from flask import render_template
from flask import Flask
from flask import jsonify
from flask_cors import CORS
from threading import Thread
import torch
from transformers import BertTokenizer
from transformers import AutoConfig
from transformers.modeling_bert import BertForQuestionAnswering
import time
import os
import csv
from src.covid_browser.fast_qa import pre_process, do_qa
from src.covid_browser.utils import read_from_file, get_data, read_json_file
import logging
# cord19q imports
from src.cord19q.models import Models
from src.cord19q.query import Query

torch.set_grad_enabled(False)
application = Flask(__name__)
application.config["JSON_SORT_KEYS"] = False  # this keeps Flask from reordering the sort order of dictionary keys
# during jsonfiy. Otherwise if you have a dict ordered by scores, Flask will serialize JSON objects in a way that the
# keys are ordered. See https://github.com/pallets/flask/issues/974
CORS(application)
logging.basicConfig(filename='logs/log.txt', level=logging.INFO)
logfile = logging.getLogger('file')
use_cuda = True
if os.environ.get('USE_GPU'):
    use_cuda = True
    print('using CUDA')
else:
    use_cuda = True

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


def threaded_task(model_name):
    load_model_(model_name, 0)


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


@application.route('/load_model/<model_name>', methods=['POST', 'GET'])
def load_model(model_name, gpu_id=0):
    return jsonify(load_model_(model_name, gpu_id))


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


@application.route('/cord19q_lookup/<query>', methods=['POST'])
def cord19q_lookup(query):
    """
    Uses the model to find the answer to the query in the currently active document.
    """
    query = query + '?'
    start = time.time()
    embeddings, db = Models.load(model_names_to_path['cord19q'])
    bert_model = cache[curr_model]['model'] if curr_model in cache else None
    bert_tokenizer = cache[curr_model]['tokenizer'] if curr_model in cache else None
    articles = cache.get("articles")

    cord19q_docs = Query.query2(embeddings, db, query, 10)
    print('query time: {0}'.format(time.time() - start))
    if cord19q_docs:
        # load the corresponding doc
        doc_info = {}
        sentences_all = []
        for k, v in cord19q_docs.items():
            source = articles.get(k)  # load the info about the article referenced by the uid
            if source:
                path = os.path.join('/home/ankur/dev/apps/ML/Covid-19', source['subset'], source['subset'], k + '.json')
                # read article text and create list of sentences
                contents = read_json_file([path])
                parsed_contents = get_data(contents)
                sentences = parsed_contents[-1].get(k)
                for s in sentences:
                    sentences_all.append(s)
                # update the doc with the content
                doc_info.update({k: (sentences, source.get('title'), source.get("abstract"),
                                     source.get('publish_time'), source.get('authors'),
                                     source.get('journal'))})
        # create spans
        spans = pre_process(sentences_all, bert_tokenizer)
        bert_answers = do_qa(spans, query, bert_tokenizer, bert_model)
        logfile.info('successfully ran do_qa on query: {0}'.format(query))
        return jsonify({'success': True, 'cordq_answers': cord19q_docs, 'bert_answers': bert_answers})
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
    articles = dict()
    with open(os.path.join('/home/ankur/dev/apps/ML/covid-papers-analysis/data', "metadata.csv"),
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
    # load the bert models in a separate thread so we don't wait for the model to load
    thread = Thread(target=threaded_task, args=('bert-base-uncased_finetuned_squad',))
    thread.daemon = True
    thread.start()
    # load_model_('bert-base-uncased_finetuned_squad')
    # run with debug = False, so that Flask doesn't attempt to autoload when the static HTML content changes. That can
    # lead to the GPU memory not getting properly cleaned. Downside is that Flask server needs to be stopped and
    # restarted every time the template (main.html) is modified.
    application.run(debug=False)
