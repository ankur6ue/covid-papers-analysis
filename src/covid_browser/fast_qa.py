# Split on whitespace so that different tokens may be attributed to their original position.
import torch
import numpy as np
from typing import Tuple
import os
import time
import psutil
import onnxruntime
import random
onnx = False
if onnx:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1 ## psutil.cpu_count(logical=True)
    ort_session = onnxruntime.InferenceSession(
        dir_path + '/data/models/bert-base-uncased_finetuned_squad.onnx', sess_options)

def decode(start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int) -> Tuple:
    """
    Take the output of any QuestionAnswering head and will generate probalities for each span to be
    the actual answer.
    In addition, it filters out some unwanted/impossible cases like answer len being greater than
    max_answer_len or answer end position being before the starting position.
    The method supports output the k-best answer through the topk argument.

    Args:
        start: numpy array, holding individual start probabilities for each token
        end: numpy array, holding individual end probabilities for each token
        topk: int, indicates how many possible answer span(s) to extract from the model's output
        max_answer_len: int, maximum size of the answer to extract from the model's output
    """
    # Ensure we have batch axis
    if start.ndim == 1:
        start = start[None]

    if end.ndim == 1:
        end = end[None]

    # Compute the score of each tuple(start, end) to be the real answer
    outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

    # Remove candidate with end < start and end - start > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
    return start, end, candidates[0, start, end]


def pre_process(sentences, tokenizer, max_query_length=64, max_seq_length=384):
    """
    Given a list of sentences, creates a list of spans. Sentences are successively added to a span until the
    number of tokens in the span (including the query and special tokens [CLS, SEP]) exceed max_seq_length. The
    sentence that causes the overflow is added to the next span.
    Tokenization for sentences is performed once and tok_to_orig index is adjusted when inference is performed
    on a query. Avoiding repeated tokenization of the document text makes inference much faster than the
    hugginsface transformers implementation
    """
    num_sentences = len(sentences)
    spans = []
    tok_to_orig_index = []  # token to original mapping
    # original:         antimicrobial resistance ..
    # orig index:       0               1
    # tokens:           anti ##micro ##bial resistance ..
    # tok_to_orig_index: 0      0       0       1..
    # orig_to_tok_index: 0              3..
    orig_to_tok_index = []  # original words to tokens mapping
    all_span_tokens = []  # all tokens in a span
    curr_num_tokens = 0
    words = []
    i = 0
    while i < num_sentences:
        sentence = sentences[i]
        i = i + 1
        words_ = sentence.split()  # split sentence into words
        for (j, token) in enumerate(words_):
            sub_tokens = tokenizer.tokenize(token)  # split words into tokens (antimicrobial: anti, ##micro, ##bial)
            curr_num_tokens = curr_num_tokens + len(sub_tokens)  # maintain a count of number of tokens so far
            # at inference time, we'll append the tokenized query (whose max length in tokens is max_seq_length)
            # and also add three extra tokens: CLS tokenized_query SEP tokenized_sentences SEP.
            if curr_num_tokens < max_seq_length - max_query_length - 3:
                # all_span_tokens records the tokens encountered so far
                orig_to_tok_index.append(len(all_span_tokens))
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(len(words))
                    all_span_tokens.append(sub_token)
                words.append(token)
            else:
                #  Adding this sentence will exceed the number of tokens allowed. Finalize span and reset
                spans.append({'orig_to_tok_index': orig_to_tok_index,
                              'tok_to_orig_index': tok_to_orig_index,
                              'all_span_tokens': all_span_tokens,
                              'words': words})
                # reset:
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_span_tokens = []
                curr_num_tokens = 0
                words = []
                # back up 1, because last sentence caused overflow and wasn't added in its entirety.
                i = i - 1
                break
    return spans


def do_qa(spans, query, tokenizer, model, device=-1, max_query_length=64, max_seq_length=384, topk=5,
          max_answer_len=15, logfile=None):
    # tokenize query, truncate if length exceeds max_query_length
    truncated_query = tokenizer.encode(query, add_special_tokens=False, max_length=max_query_length)
    len_truncated_query = len(truncated_query)
    input_ids = torch.LongTensor().to(model.device)
    attention_mask = torch.LongTensor().to(model.device)
    token_type_ids = torch.LongTensor().to(model.device)
    print("batch size: {0}".format(len(spans)))
    if logfile:
        logfile.info("batch size: {0}".format(len(spans)))
    # randomly select 45 spans from the span list. This bounds max runtime and prevents potential out-of-memory
    # errors, at the expense of potentially missing some promising spans.
    if len(spans) > 45:
        spans = random.sample(spans, 45)

    for span in spans:
        # we'll perform inference on each span
        span_tokens = span['all_span_tokens']
        # appends the truncated query to the span tokens and adds special token and padding. By construction of
        # spans, overflow can't occur.
        # encoded_dict['input_ids'] looks like this:
        # CLS <query tokens> SEP <sentence tokens> SEP PADDING.
        # Padding is added so total length of the encoded_dict['input_ids'] = max_seq_length
        encoded_dict = tokenizer.encode_plus(
            truncated_query,
            span_tokens,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            pad_to_max_length=True,
            stride=0,
            truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            return_token_type_ids=True,
        )

        # Set the CLS index to '0'
        # offset marks the position where the document tokens begin. This will be used to adjust
        # the tok_to_orig_index so we can map back to the original words in the document
        span.update({"input_ids": encoded_dict['input_ids'],
                     "attention_mask": encoded_dict['attention_mask'],
                     "token_type_ids": encoded_dict['token_type_ids'],
                     "offset": len_truncated_query + 2})

        # add mask for masking results
        # copying logic from transformers
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.array(span["token_type_ids"])

        p_mask = np.minimum(p_mask, 1)

        if tokenizer.padding_side == "right":
            # Limit positive values to one
            p_mask = 1 - p_mask

        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1
        # Set the CLS index to '0'
        p_mask[cls_index] = 0
        span.update({"p_mask": p_mask.tolist()})

        input_ids = torch.cat(
            (input_ids, torch.tensor(encoded_dict['input_ids'], dtype=torch.long).unsqueeze(0).to(model.device)), 0)
        attention_mask = torch.cat((attention_mask,
                                    torch.tensor(encoded_dict['attention_mask'], dtype=torch.long).unsqueeze(0).to(
                                        model.device)), 0)
        token_type_ids = torch.cat((token_type_ids,
                                    torch.tensor(encoded_dict['token_type_ids'], dtype=torch.long).unsqueeze(0).to(
                                        model.device)), 0)

    fw_args = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
    if onnx is False:
        with torch.no_grad():
            start_ = time.time()
            start, end = model(**fw_args)
            print('BERT execution time: {0}'.format(time.time() - start_))
        start, end = start.cpu().numpy(), end.cpu().numpy()
    else:
        # start = time.time()
        input_ids_numpy = input_ids.numpy()
        attention_mask_numpy = attention_mask.numpy()
        token_type_ids_numpy = token_type_ids.numpy()
        # numpy conversion time is insignificant
        # print('numpy conversion time: {0}'.format(time.time() - start))
        ort_inputs = {ort_session.get_inputs()[0].name: input_ids_numpy,
                      ort_session.get_inputs()[1].name: attention_mask_numpy,
                      ort_session.get_inputs()[2].name: token_type_ids_numpy}
        ort_outs = ort_session.run(["output_start_logits", "output_end_logits"], ort_inputs)
        start = ort_outs[0]
        end = ort_outs[1]
    scores = []
    for (feature, start_, end_) in zip(spans, start, end):
        # Normalize logits and spans to retrieve the answer
        start_ = np.exp(start_) / np.sum(np.exp(start_))
        end_ = np.exp(end_) / np.sum(np.exp(end_))

        # Mask padding and question
        start_, end_ = (
            start_ * np.abs(np.array(feature["p_mask"]) - 1),
            end_ * np.abs(np.array(feature["p_mask"]) - 1),
        )

        # TODO : What happens if not possible
        # Mask CLS
        start_[0] = end_[0] = 0
        starts, ends, scores_ = decode(start_, end_, topk=1, max_answer_len=max_answer_len)
        if scores_[0] > 0.05:
            scores.append({"starts": starts[0], "ends": ends[0], "scores": scores_[0], "feature": feature})
    # sort scores
    scores = sorted(scores, key=lambda x: -x['scores'])
    answers = []
    for i in range(0, min(topk, len(scores))):
        top_score_feature = scores[i]['feature']
        offset = top_score_feature['offset']
        start = scores[i]['starts'] - offset  # map back to tok_to_orig_index space
        end = scores[i]['ends'] - offset
        score = scores[i]['scores']
        words = top_score_feature['words']
        tok_to_orig_index = top_score_feature['tok_to_orig_index']
        start_word_index = tok_to_orig_index[start]
        end_word_index = tok_to_orig_index[end]
        answer = " ".join([words[idx] for idx in range(start_word_index, end_word_index + 1)])
        context = " ".join([words[idx] for idx in range(0, len(words))])
        # we'll return the context as well as the answer
        answers.append({
            'answer': answer,
            'start': start_word_index,
            'end': end_word_index,
            'score': score,
            'context': context
        })
    return answers

# tokenize query:
# truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
# sequence = build_inputs_with_special_tokens(ids, context_ids)
