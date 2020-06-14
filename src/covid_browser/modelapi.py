import concurrent.futures
from flask.views import MethodView
from flask import request, jsonify
import time
import logging
import os
from .errors import BadQuery, ServerError
from .globals import use_cuda, cache, gpu_state, cord19q_path, base_path, curr_model, num_gpus
from .validation import Validate
from .utils import read_larger_json_file, get_data
from .log_requests import log_request_info
from src.cord19q.query import Query
from .fast_qa import pre_process, do_qa
from marshmallow import Schema, fields, ValidationError


class ModelAPI(MethodView):
    """
    Implements the bulk of the search functionality.
    1. Performs basic sanity checks on the query string
    2. Runs the query through the cord19q search to obtain top-n sentences and documents
    3. Loads the full text of the top-n documents, parses the full text into sentences and constructs BERT spans
    4. Runs the BERT model on the spans to produce excerpts.
    5. top-n sentences, excerpts and document info (published date, URL) etc are returned back to the client
    """
    def post(self, query):

        logfile = logging.getLogger('file')
        topn_bert = 7
        """
        Uses the model to find the answer to the query in the currently active document.
        """
        embeddings = cache['cord19q']['embeddings']
        try:
            # do basic sanity test on the query. Non zero length, < 20 words, only alphanumeric characters
            status, msg = Validate.query_validate(query, embeddings)
        except Exception as err:
            logfile.exception('Query validation exception')
            raise ServerError(msg, status_code=500)

        if not status:
            raise BadQuery(msg, status_code=412)  # 412: Http code for pre-condition failed

        # print and log basic info about the request
        log_request_info(request)
        query = query + '?'
        print("query: {0}".format(query))
        start = time.time()
        # wait until GPU is available.
        while gpu_state.busy:
            time.sleep(1)
            if time.time() - start > 6:
                raise ServerError('server busy, please try again later', status_code=503)

        # set GPU busy state so multiple requests can't cause out-of-memory state
        gpu_state.busy = True

        try:
            articles = cache.get("articles")
            # run embeddings look up to retrieve best matching sentences and the documents containing those sentences
            cord19_results = Query.query2(cord19q_path, embeddings, query, 20)
            print('Num docs found: {1}, query time: {0}'.format(time.time() - start, len(cord19_results)))

            # Now run BERT on these results to identify spans within the body text of the docs
            if cord19_results:
                # load the body text of the corresponding documents and return a list of all sentences
                start = time.time()
                # sequential execution actually happens to be faster, probably due to thread creation overhead..
                sentences_all, cord19_results = process_results_sequential(cord19_results, articles, topn_bert)
                print('document parsing time: {0}'.format(time.time() - start))

                bert_model = cache[curr_model.model]['model'] if curr_model.model in cache else None
                bert_tokenizer = cache[curr_model.model]['tokenizer'] if curr_model.model in cache else None

                # Create spans (a span is a group of sentences such that the total number of tokens in a span
                # is less than a threshold). BERT operates on a batch of spans for efficient execution.
                start = time.time()
                spans = pre_process(sentences_all, bert_tokenizer)
                print('span creation time: {0}'.format(time.time() - start))

                # Now do QA
                bert_answers = do_qa(spans, query, bert_tokenizer, bert_model, logfile)
                print('finished doing qa, bert_answers: {0}'.format(len(bert_answers)))
                logfile.info('successfully ran do_qa on query: {0}'.format(query))
                gpu_state.busy = False
                return jsonify({'success': True, 'cordq_answers': cord19_results, 'bert_answers': bert_answers})
            else:  ## no docs found
                gpu_state.busy = False
                return jsonify({'success': False, 'msg': 'no answers found'})

        except Exception as e:
            logfile.exception('Error running query:{0} Exception {1} '.format(query, e))
            print('Error running query:{0} Exception {1} '.format(query, e))
            gpu_state.busy = False  # otherwise if there is an error, GPU state will never be reset
            raise ServerError('error running query', status_code=500)


def process_results_sequential(cord19_results, articles, topn):
    """
    Load the full text corresponding to the top-n documents, tokenize the contents into sentences and return
    the sentences along with other document related info such as title, publication date, document url etc.
    Turns out that doing this sequentially is faster than spliting the processing over multiple threads.
    """
    sentences = []
    num = 0
    for k, v in cord19_results.items():
        source = articles.get(k)  # load the info about the article referenced by the uid
        if source:
            path_pmc = os.path.join(base_path, 'data/cord19q/current', source['pmc_json_files'])
            path_pdf = os.path.join(base_path, 'data/cord19q/current', source['pdf_json_files'])
            contents = read_larger_json_file([path_pmc, path_pdf])
            for s in _process(contents):
                sentences.append(s)
            url = source['url']
            tup = cord19_results[k]
            # currently: scores, title, published, publication
            cord19_results[k] = (tup[0], tup[1], tup[2], tup[3], url)
            num = num + 1
            if num > topn:
                break
    return sentences, cord19_results


def process_results_threaded(cord19_results, articles, topn):
    doc_info = {}
    sentences = []
    num = 0
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for k, v in cord19_results.items():
            source = articles.get(k)  # load the info about the article referenced by the uid
            if source:
                path_pmc = os.path.join(base_path, 'data/cord19q/current', source['pmc_json_files'])
                path_pdf = os.path.join(base_path, 'data/cord19q/current', source['pdf_json_files'])
                contents = read_larger_json_file([path_pmc, path_pdf])
                futures.append(executor.submit(_process, contents))
                url = source['url']
                tup = cord19_results[k]
                # currently: scores, title, published, publication
                cord19_results[k] = (tup[0], tup[1], tup[2], tup[3], url)
                num = num + 1
                if num > topn:
                    break

        for future in futures:
            for s in future.result():
                sentences.append(s)
        return sentences, cord19_results


def _process(contents):
    # read article text and create list of sentences
    # contents = read_json_file([path])
    sentences = []
    if contents:
        parsed_contents = get_data(contents)
        sentences_ = list(parsed_contents[-1].values())[0]
        for s in sentences_:
            # make sure sentence only contains ascii characters. Otherwise the BERT tokenizer can take
            # forever to tokenize
            en = all(ord(c) < 128 for c in s)
            if en:
                sentences.append(s)
    return sentences
