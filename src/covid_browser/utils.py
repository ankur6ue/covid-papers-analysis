""" Provides utility methods used in scripts."""

import csv
import json

import os
from os import path

from gensim.summarization.textcleaner import get_sentences
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, \
    strip_multiple_whitespaces, strip_numeric, \
    remove_stopwords, strip_short, strip_non_alphanum

CUSTOM_FILTERS = [
    lambda x: x.lower(), strip_tags, strip_punctuation,
    strip_multiple_whitespaces, strip_numeric, strip_non_alphanum,
    remove_stopwords, strip_short
]


# for reading sentences. Returns list of words in a sentence
def read_from_file(filename, delimiter='\t'):
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        data = list(reader)
        return data
    return None


def clean_references_and_citations(item):
    cite_spans = item['cite_spans']
    ref_spans = item['ref_spans']
    text = item['text']
    for cite_span in cite_spans:
        strip_text = cite_span['text']
        # can't use string.strip, because it only strips characters at the beginning and end of a string!
        text = text.replace(strip_text, '')
    for ref_span in ref_spans:
        strip_text = ref_span['text']
        # can't use string.strip, because it only strips characters at the beginning and end of a string!
        text = text.replace(strip_text, '')
    return text

def consolidate_text(blob):
    # convert: "check out these cool references [1][2]" ->
    # "check out these cool references
    consolidated_text = ""
    for item in blob:
        consolidated_text = consolidated_text + clean_references_and_citations(item)

    return consolidated_text


def clean_text(blob):
    # convert: "check out these cool references [1][2]" ->
    # "check out these cool references
    paragraphs = []
    for item in blob:
        paragraphs.append(clean_references_and_citations(item))

    return paragraphs

# parses the article json, applies text filters to the blocks of body text and returns dictionaries
# for titles, abstracts, authors and a list of sentences
def get_data(articles):
    # document id -> abstract
    id2abstract = {}
    # document id -> title
    id2title = {}
    # document id -> authors
    id2authors = {}
    # list of pre-processed sentences
    sentences_processed = []
    # document id -> list of original sentences, converted to lower case
    id2sentences = {}
    for article in articles:
        id = article['paper_id']
        title = article['metadata']['title']
        authors = article['metadata']['authors']
        bodytext = clean_text(article['body_text'])
        abstract = consolidate_text(article['abstract'])
        sentences = []
        for para in bodytext:
            for sentence in get_sentences(para):
                # remove extra whitespaces behind period
                # "how are you  ." becomes "how are you."
                if (sentence[-1] == '.'):
                    sentence = sentence[:-1].rstrip()
                    sentence = sentence + '.'
                sentences_processed.append(preprocess_string(sentence, CUSTOM_FILTERS))
                sentences.append(sentence.lower())
            # append a newline at the end of the last sentence to indicate paragraph break
            if len(sentences) is not 0:
                last_sentence = sentences[-1]
                last_sentence = last_sentence + '\n'
                sentences[-1] = last_sentence

        id2sentences.update({id: sentences})
        # should probably apply custom filters to the abstract as well..
        id2abstract.update({id: abstract})
        id2title.update({id: title})
        id2authors.update({id: authors})

    return [id2abstract, id2title, id2authors, sentences_processed, id2sentences]


# Reads the contents of json files in files list
def read_json_file(files):
    contents = []
    for file in files:
        with open(file) as json_data:
            data = json.load(json_data)
            contents.append(data)
    return contents


def clean_references_and_citations(item):
    cite_spans = item['cite_spans']
    ref_spans = item['ref_spans']
    text = item['text']
    for cite_span in cite_spans:
        strip_text = cite_span.get('text')
        # can't use string.strip, because it only strips characters at the beginning and end of a string!
        if strip_text:
            text = text.replace(strip_text, '')
    for ref_span in ref_spans:
        strip_text = ref_span.get('text')
        # can't use string.strip, because it only strips characters at the beginning and end of a string!
        if strip_text:
            text = text.replace(strip_text, '')
    return text

def clean_text(blob):
    # convert: "check out these cool references [1][2]" ->
    # "check out these cool references
    paragraphs = []
    if blob:
        for item in blob:
            paragraphs.append(clean_references_and_citations(item))

    return paragraphs


# parses the article json, applies text filters to the blocks of body text and returns dictionaries
# for titles, abstracts, authors and a list of sentences
def get_data(articles):
    # document id -> abstract
    id2abstract = {}
    # document id -> title
    id2title = {}
    # document id -> authors
    id2authors = {}
    # list of pre-processed sentences
    sentences_processed = []
    # document id -> list of original sentences, converted to lower case
    id2sentences = {}
    for article in articles:
        id = article['paper_id']
        title = article['metadata'].get('title')
        authors = article['metadata'].get('authors')
        bodytext = clean_text(article.get('body_text'))
        abstract = clean_text(article.get('abstract'))
        sentences = []
        for para in bodytext:
            for sentence in get_sentences(para):
                # remove extra whitespaces behind period
                # "how are you  ." becomes "how are you."
                if (sentence[-1] == '.'):
                    sentence = sentence[:-1].rstrip()
                    sentence = sentence + '.'
                sentences_processed.append(preprocess_string(sentence, CUSTOM_FILTERS))
                sentences.append(sentence.lower())
            # append a newline at the end of the last sentence to indicate paragraph break
            if len(sentences) is not 0:
                last_sentence = sentences[-1]
                last_sentence = last_sentence + '\n'
                sentences[-1] = last_sentence

        id2sentences.update({id: sentences})
        # should probably apply custom filters to the abstract as well..
        id2abstract.update({id: abstract})
        id2title.update({id: title})
        id2authors.update({id: authors})

    return [id2abstract, id2title, id2authors, sentences_processed, id2sentences]


# Reads the contents of json files in files list
def read_json_file(files):
    contents = []
    for file in files:
        with open(file) as json_data:
            data = json.load(json_data)
            contents.append(data)
    return contents


# Read the larger of the two json files
def read_larger_json_file(files):
    contents = []
    max_size = 0
    max_idx = 0
    idx = 0
    for file in files:
        if path.exists(file):
            stat = os.stat(file)
            if stat.st_size > max_size:
                max_size = stat.st_size
                max_idx = idx
        idx = idx + 1
    if max_size == 0:
        return None  # No files found
    with open(files[max_idx]) as json_data:
        data = json.load(json_data)
        contents.append(data)
    return contents

'''
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
        
@application.route('/load_model/<model_name>', methods=['POST', 'GET'])
def load_model(model_name, gpu_id=0):
    return jsonify(load_model_(model_name, gpu_id))

@application.route('/')
@application.route('/main/')
@application.route('/main/<name>')
def main(name=None):
    """
    Loads the main application front-end
    """
    return render_template('main.html', serv_addr=serv_addr, name=name)


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

'''