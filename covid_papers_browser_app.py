from flask import render_template
from flask import Flask
from flask import jsonify
from flask_cors import CORS
from threading import Thread
import torch
from transformers import BertTokenizer
from transformers.modeling_bert import BertForQuestionAnswering
import time
import os
import csv
from src.covid_browser.fast_qa import pre_process, do_qa
from src.covid_browser.utils import read_from_file, get_data, read_json_file, is_proper
import logging
# cord19q imports
from src.cord19q.models import Models
from src.cord19q.query import Query
import numpy as np
torch.set_grad_enabled(False)
application = Flask(__name__)
application.config["JSON_SORT_KEYS"] = False  # this keeps Flask from reordering the sort order of dictionary keys
# during jsonfiy. Otherwise if you have a dict ordered by scores, Flask will serialize JSON objects in a way that the
# keys are ordered. See https://github.com/pallets/flask/issues/974
CORS(application)
logging.basicConfig(filename='logs/log.txt', level=logging.INFO)
logfile = logging.getLogger('file')
use_cuda = True
print('USE_GPU: {0}'.format(os.environ.get('USE_GPU')))
if os.environ.get('USE_GPU') == '1':
    use_cuda = True
    print('using CUDA')
else:
    use_cuda = False

# Global state that records if GPU is currently processing a request
is_gpu_busy = False
serv_addr = os.environ.get('SERV_ADDR') if os.environ.get('SERV_ADDR') else 'http://127.0.0.1:5000'
print('server address: {0}'.format(serv_addr))
logfile.info('server address: {0}'.format(serv_addr))

cache = {}
dir_path = os.path.dirname(os.path.realpath(__file__))
model_names_to_path = \
    {"bert-large-uncased-whole-word-masking-finetuned-squad":
         os.path.join(dir_path, 'data/models/bert-large-uncased-whole-word-masking-finetuned-squad'),
     "bert-base-uncased_finetuned_squad":
         os.path.join(dir_path, 'data/models/bert-base-uncased_finetuned_squad'),
     "cord19q": os.path.join(dir_path, 'data/cord19q')
     }

curr_model = None


def export_to_onnx(model, onnx_file_name):
    model.eval()
    eval_batch_size = 1
    max_seq_length = 384
    device = torch.device("cpu")
    model.to(device)
    dummy_input = torch.ones((1, 384), dtype=torch.long)

    torch.onnx.export(model,
                      (dummy_input, dummy_input, dummy_input),
                      onnx_file_name,
                      input_names=["input_ids", "input_mask", "segment_ids"],
                      verbose=True,
                      output_names=["output_start_logits", "output_end_logits"],
                      do_constant_folding=True, opset_version=11,
                      dynamic_axes=({"input_ids": {0: "batch_size"}, "input_mask": {0: "batch_size"},
                                     "segment_ids": {0: "batch_size"},
                                     "output_start_logits": {0: "batch_size"}, "output_end_logits": {0: "batch_size"}}))


def threaded_task1(model_name):
    load_model_(model_name, 0)
    model = cache.get(model_name)['model']
    path = model_names_to_path[model_name]
    onnx_file_name = path + '.onnx'
    #export_to_onnx(model, onnx_file_name)
    logfile.info('finished loaded BERT model')
    print('finished loaded BERT model')


def threaded_task2():
    if not cache.get("cord19q"):
        embeddings, db = Models.load(model_names_to_path['cord19q'])
        cache.update({'cord19q': {'embeddings': embeddings, 'db': db}})
        logfile.info('finished loaded cord19q data')
        print('finished loaded cord19q data')


def load_model_(model_name, gpu_id=0):
    """
       Loads the model and associated tokenizer using the transformers library and adds the model to the cache
       """
    global curr_model
    start = time.time()
    # check if model has already been loaded
    if not cache or model_name not in cache:
        # if not already loaded, check if we have a path for this model
        model_path = model_names_to_path.get(model_name)
        if not model_path:
            # we don't have a path to load the model artifacts, log error anr return
            # HTTP Status Code automatically set to 200 and Content-Type header to application/json
            logfile.info('path not found for model: {0}'.format(model_name))
            return {'success': False, 'msg': 'model path not found'}
        else:
            logfile.info('found path for: {0}'.format(model_name))
            try:
                tokenizer = BertTokenizer(
                    **{'vocab_file': os.path.join(model_path, 'vocab.txt'), 'max_len': 512, 'do_lower_case': True})
                tokenizer.unique_added_tokens_encoder.update(set(tokenizer.all_special_tokens))
                model = BertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=model_path)
                # model.half() not implemented on CPUs!!
                if use_cuda and torch.cuda.is_available():
                    model = model.cuda(gpu_id)
                cache.update({model_name: {'model': model, 'tokenizer': tokenizer}})
                logfile.info('model: {0} successfully loaded'.format(model_name))
            except:
                logfile.info('Found path but error loading model: {0}'.format(model_name))
                return {'success': False, 'msg': 'error loading model'}
    else:  # we already have the model in the cache, just return
        model = cache[model_name]
    curr_model = model_name
    return {'success': True, 'load_time': time.time() - start}


@application.route('/cord19q_lookup/<query>', methods=['POST'])
def cord19q_lookup(query):
    global is_gpu_busy
    """
    Uses the model to find the answer to the query in the currently active document.
    """
    try:
        # do basic sanity test on the query. Non zero length, < 20 words, only alphanumeric characters
        if not is_proper(query):
            return {'success': False, 'msg': 'Query must be non-zero length and contain fewer than 100 alphanumeric \
                                             characters (with the exception of ? and -)'}
        query = query + '?'
        print("query: {0}".format(query))
        start = time.time()
        # wait until GPU is available.
        while is_gpu_busy:
            time.sleep(1)
            if time.time() - start > 6:
                return {'success': False, 'msg': 'server busy, please try again later'}
        # set GPU busy state so multiple requests can't cause out-of-memory state
        is_gpu_busy = True
        # embeddings, db = Models.load(model_names_to_path['cord19q'])
        embeddings = cache['cord19q']['embeddings']
        bert_model = cache[curr_model]['model'] if curr_model in cache else None
        bert_tokenizer = cache[curr_model]['tokenizer'] if curr_model in cache else None
        articles = cache.get("articles")
        cord19q_docs = Query.query2(model_names_to_path['cord19q'], embeddings, query, 10)
        print('Num docs found: {1}, query time: {0}'.format(time.time() - start, len(cord19q_docs)))
        if cord19q_docs:
            # load the corresponding doc
            doc_info = {}
            sentences_all = []
            topn_bert = 5
            num = 0
            for k, v in cord19q_docs.items():
                source = articles.get(k)  # load the info about the article referenced by the uid
                if source:
                    # add url to the tuples in cord19q
                    url = source['url']
                    tup = cord19q_docs[k]
                    # currently: scores, title, published, publication
                    tup_ = (tup[0], tup[1], tup[2], tup[3], url)
                    cord19q_docs[k] = tup_
                    path = os.path.join(dir_path, 'data', source['subset'], source['subset'], k + '.json')
                    # read article text and create list of sentences
                    contents = read_json_file([path])
                    parsed_contents = get_data(contents)
                    sentences = parsed_contents[-1].get(k)
                    for s in sentences:
                        # make sure sentence only contains ascii characters. Otherwise the BERT tokenizer can take
                        # forever to tokenize
                        en = all(ord(c) < 128 for c in s)
                        if en:
                            sentences_all.append(s)
                    # update the doc with the content
                    doc_info.update({k: (sentences, source.get('title'), source.get("abstract"),
                                         source.get('publish_time'), source.get('authors'),
                                         source.get('journal'))})
                    # The idea is to only construct spans for the top 5 docs. We do this because running BERT model
                    # is rather time consuming. TBD: In a multi-gpu setup, would be good to split up BERT inference
                    # across multiple GPUs!
                    num = num + 1
                    if num > 5:
                        break
            # create spans
            # TBD: Parallelize this across multiple processes.
            spans = pre_process(sentences_all, bert_tokenizer)
            bert_answers = do_qa(spans, query, bert_tokenizer, bert_model, logfile)
            print('finished doing qa, bert_answers: {0}'.format(len(bert_answers)))
            logfile.info('successfully ran do_qa on query: {0}'.format(query))
            is_gpu_busy = False
            return jsonify({'success': True, 'cordq_answers': cord19q_docs, 'bert_answers': bert_answers})
        else:
            is_gpu_busy = False
            return jsonify({'success': False, 'msg': 'no answers found'})
    except Exception as e:
        logfile.info('Error running query:{0} Exception {1} '.format(query, e))
        print('Error running query:{0} Exception {1} '.format(query, e))
        is_gpu_busy = False  # otherwise if there is an error, GPU state will never be reset
        return {'success': False, 'msg': 'error running query'}


if __name__ == '__main__':
    print("CUDA availability: {0}".format(torch.cuda.is_available()))
    articles = dict()

    with open(os.path.join(dir_path, 'data', "metadata.csv"),
              mode="r") as csvfile:
        csv_dict = csv.DictReader(csvfile)
        for row in csv_dict:
            articles.update({row['sha']: {'subset': row['full_text_file'],
                                          "title": row["title"],
                                          "abstract": row["abstract"],
                                          "publish_time": row["publish_time"],
                                          "authors": row["authors"],
                                          "journal": row["journal"],
                                          "url": row["url"]}})
        cache.update({'articles': articles})
        print('loaded articles')
    # load the bert models in a separate thread so we don't wait for the model to load
    if use_cuda and torch.cuda.is_available(): # use large bert model if cuda available, otherwise use small
        thread = Thread(target=threaded_task1, args=('bert-large-uncased-whole-word-masking-finetuned-squad',))
        print('using large BERT model')
    else:
        thread = Thread(target=threaded_task1, args=('bert-base-uncased_finetuned_squad',))
        print('using small BERT model')
    thread.daemon = True
    thread.start()
    Thread(target=threaded_task2).start()
    # for local debuggins without starting the web server
    time.sleep(5);
    cord19q_lookup("covid-19?")
    # load_model_('bert-base-uncased_finetuned_squad')
    # run with debug = False, so that Flask doesn't attempt to autoload when the static HTML content changes. That can
    # lead to the GPU memory not getting properly cleaned. Downside is that Flask server needs to be stopped and
    # restarted every time the template (main.html) is modified.
    application.run(debug=False, host='0.0.0.0')
