""" Provides utility methods used in scripts."""

import csv
import json
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

