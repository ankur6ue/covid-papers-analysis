from flask import render_template
from flask import Flask
from flask import jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer
from transformers import AutoConfig
from transformers.modeling_bert import BertForQuestionAnswering
import time
import os
from src.covid_browser.fast_qa import pre_process, do_qa
from src.covid_browser.utils import read_from_file
import logging

torch.set_grad_enabled(False)
application = Flask(__name__)
CORS(application)
logging.basicConfig(filename='logs/log.txt', level=logging.INFO)
logfile = logging.getLogger('file')
use_cuda = False
if os.environ.get('USE_GPU'):
    use_cuda = True
    print('using CUDA')

cache = {}
dir_path = os.path.dirname(os.path.realpath(__file__))
model_names_to_path = \
    [
        {'model_name': "bert-large-uncased-whole-word-masking-finetuned-squad",
         'path': os.path.join(dir_path, 'data/models/bert-large-uncased-whole-word-masking-finetuned-squad')},
        {'model_name': "bert-base-uncased_finetuned_squad",
         'path': os.path.join(dir_path, 'data/models/bert-base-uncased_finetuned_squad')}
    ]
curr_model = None


@application.route('/load_model/<model_name>', methods=['POST', 'GET'])
def load_model(model_name, gpu_id=0):
    """
    Loads the model and associated tokenizer using the transformers library and adds the model to the cache
    """
    global curr_model
    start = time.time()
    if not cache or model_name not in cache:
        # look up path
        model_path = None
        for model in model_names_to_path:
            if model['model_name'] == model_name:
                logfile.info('found path for: {0}'.format(model_name))
                model_path = model['path']
                break

        if model_path is None:
            # HTTP Status Code automatically set to 200 and Content-Type header to application/json
            logfile.info('path not found for model: {0}'.format(model_name))
            return jsonify({'success': False, 'msg': 'model path not found'})
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
            logfile.info('error loading model: {0}'.format(model_name))
            return jsonify({'success': False, 'msg': 'error loading model'})
    else:
        model = cache[model_name]
    curr_model = model_name
    return jsonify({'success': True, 'load_time': time.time() - start})


@application.route('/main/')
@application.route('/main/<name>')
def main(name=None):
    """
    Loads the main application front-end
    """
    return render_template('main.html', name=name)


@application.route('/get_abstract/<doc_id>', methods=['POST', 'GET'])
def get_abstract(doc_id):
    """
    For a given doc_id, returns the corresponding abstract by looking up the doc_id in abstracts.csv
    """
    file_path = os.path.join(dir_path, 'data/abstracts.csv')
    abstracts = read_from_file(file_path)
    if abstracts:
        logfile.info('Reading abstracts')
        cache['curr_docid'] = None
        for abstract in abstracts:
            if abstract[0] == doc_id:
                if abstract[1]:
                    cache['curr_docid'] = doc_id
                    return jsonify({'success': True, 'context': abstract[1]})
                else:
                    return jsonify({'success': False, 'msg': 'no abstract text'})
        return jsonify({'success': False, 'msg': 'no matching doc id'})
    else:
        logfile.info('Error reading abstracts')
        return jsonify({'success': False, 'msg': 'error loading abstracts.csv'})

@application.route('/get_titles/', methods=['POST', 'GET'])
def get_titles():
    """
    Loads the doc_id to titles mapping and returns the first 50 titles. Also loads the sentences csv file and
    makes a dictionary of doc_id and doc_content key value pairs.
    The sentences.csv file is a list of sentences with the following format:
    doc_id1: sentence1
    doc_id1: sentence2
    doc_id1: sentence3
    ...
    doc_id2: sentence1
    doc_id2: sentence2

    To find the answer for a given question in a document, we must first consolidate all the sentences in that document.
    This is done in make_docid2sentences_dict function
    """
    file_path = os.path.join(dir_path, 'data/titles.csv')
    titles = read_from_file(file_path)
    if titles:
        logfile.info('Reading titles')
        # also make a dictionary of sentences in the doc_id corresponding to that title
        make_docid2sentences_dict()
        return jsonify({'success': True, 'titles': titles[0:50]})
    else:
        logfile.info('Error reading titles')
        return jsonify({'success': False, 'msg': 'could not find titles'})


@application.route('/run/<query>', methods=['POST'])
def run(query):
    """
    Uses the model to find the answer to the query in the currently active document.
    """
    query = query + '?'
    # if no current model, tokenizer or active doc, return success: False
    model = cache[curr_model]['model'] if curr_model in cache else None
    tokenizer = cache[curr_model]['tokenizer'] if curr_model in cache else None
    curr_docid = cache.get('curr_docid')
    if model is None or tokenizer is None or curr_docid is None:
        logfile.info('Error loading model/tokenizer/current document during question-answering')
        return jsonify({'success': False, 'msg': 'model and/or tokenizer and/or document not loaded'})
    # check if we already created the spans
    docid2spans = cache.get('docid2spans')
    spans = None if docid2spans is None else docid2spans.get(curr_docid)
    if spans is None:  # spans for this doc not found, let's create them
        logfile.info('Creating spans for document: {0}'.format(curr_docid))
        docid2sentences = cache.get('docid2sentences')
        if docid2sentences is None:
            logfile.info('error loading doc2sentences dictionary')
            jsonify({'success': False, 'msg': 'error loading doc2sentences dictionary'})
        # get will return None if key doesn't exist
        sentences = docid2sentences.get(curr_docid)
        if sentences is None:
            logfile.info('sentences corresponding to the doc {0} not loaded'.format(curr_docid))
            return jsonify({'success': False, 'msg': 'sentences corresponding to the current doc not loaded'})
        # create spans
        spans = pre_process(sentences, tokenizer)
        # cache the spans for faster lookup next time
        cache['docid2spans'].update({curr_docid: spans})
        logfile.info('successfully created spans for document {0}'.format(curr_docid))
    answers = do_qa(spans, query, tokenizer, model)
    logfile.info('successfully ran do_qa on query: {0}'.format(query))
    if answers:
        return jsonify({'success': True, 'answers': answers})
    return jsonify({'success': False, 'msg': 'no answers found'})


def make_docid2sentences_dict():
    file_path = os.path.join(dir_path, 'data/sentences.csv')
    sentences = read_from_file(file_path)
    if sentences:
        logfile.info('Reading sentences')
        docid2sentences = {}
        # sentences is a list of lists. The inner list has two elements - first element is the doc id and the
        # second element is a sentence. We want to consolidate all the sentences belonging to a doc and make a dict
        # so we can easily pull up all sentences in a doc.
        curr_doc_id = sentences[0][0]  # doc_id for the first element
        curr_doc_data = []
        curr_doc_data.append(sentences[0][1])  # corresponding sentence
        idx = 1
        while idx < len(sentences):
            if sentences[idx][0] == curr_doc_id:
                curr_doc_data.append(sentences[idx][1])  # keep appending sentences until the doc_id changes.
            else:
                # we have a new doc id now. So add all sentences of the previous doc to the dictionary and reset
                docid2sentences.update({curr_doc_id: curr_doc_data})
                curr_doc_data = []
                curr_doc_id = sentences[idx][0]
            idx = idx + 1
        # save to cache
        cache['docid2sentences'] = docid2sentences
        cache['docid2spans'] = {}  # create this dictionary here, we'll be needing it later
    else:
        logfile.info('Error reading sentences')


if __name__ == '__main__':
    # run with debug = False, so that Flask doesn't attempt to autoload when the static HTML content changes. That can
    # lead to the GPU memory not getting properly cleaned. Downside is that Flask server needs to be stopped and
    # restarted every time the template (main.html) is modified.
    application.run(debug=False)
