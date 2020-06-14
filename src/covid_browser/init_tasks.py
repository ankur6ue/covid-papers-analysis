from threading import Thread
import concurrent.futures
import time
import csv
import os
import logging
import torch
from transformers import BertTokenizer
from transformers.modeling_bert import BertForQuestionAnswering
from .globals import use_cuda, cache, cord19q_path, base_path, bert_large_path, bert_small_path, curr_model, num_gpus
from src.cord19q.models import Models
from copy import deepcopy
import sqlite3


def task4():
    """
    Queries the number of sentences, number of articles etc. from the sections and articles table and caches the info
    """
    logfile = logging.getLogger('file')
    try:
        dbfile = os.path.join(base_path, 'data/cord19q/current', "articles.sqlite")
        db = sqlite3.connect(dbfile)
        cur = db.cursor()
        # sentences corresponding to Covid-19 tagged docs that aren't fragment or question
        num_relevant_sentences = cur.execute("SELECT COUNT(id) FROM sections WHERE tags is not null AND " +
                                             "(labels is null or labels NOT IN ('FRAGMENT', 'QUESTION'))").fetchall()[0]
        num_all_sentences = cur.execute("SELECT COUNT(id) FROM sections").fetchall()[0]
        num_all_articles = cur.execute("SELECT COUNT(id) FROM articles").fetchall()[0]
        cache.update({'num_all_sentences': num_all_sentences})
        cache.update({'num_all_articles': num_all_articles})
        cache.update({'num_relevant_sentences': num_relevant_sentences})
        return True
    except Exception as e:
        logfile.exception(e.args)
        return False


def task3():
    """
    Reads the metadata.csv file that contains top level info about papers/articles in the covid-19 dataset. Parses
    the csv and stores article info in a dictionary with the sha as the key.
    """
    articles = dict()
    with open(os.path.join(base_path, 'data/cord19q/current', "metadata.csv"),
              mode="r") as csvfile:
        csv_dict = csv.DictReader(csvfile)
        num_articles = 0
        num_pdf_articles = 0
        num_pmc_articles = 0
        for row in csv_dict:
            articles.update({row['sha']: {'source_x': row['source_x'],
                                          'pdf_json_files': row['pdf_json_files'],
                                          'pmc_json_files': row['pmc_json_files'],
                                          "title": row["title"],
                                          "abstract": row["abstract"],
                                          "publish_time": row["publish_time"],
                                          "authors": row["authors"],
                                          "journal": row["journal"],
                                          "url": row["url"]}})

            num_articles = num_articles + 1
            if row['pdf_json_files']:
                num_pdf_articles = num_pdf_articles + 1
            if row['pmc_json_files']:
                num_pmc_articles = num_pmc_articles + 1

        cache.update({'articles': articles})
        print('loaded {0} articles'.format(num_articles))
        print('PMC articles: {0}'.format(num_pmc_articles))
        print('PDF articles: {0}'.format(num_pdf_articles))
        return True


def task1(model_name, model_path):
    """
    Losds the BERT models
    """
    start = time.time()
    status = load_model_(model_name, model_path)
    if status:
        print('{0} loaded in {1} sec'.format(model_name, time.time() - start))
    return status


def task2():
    """
    Loads the word embeddings (can take some time) and articles and sections table
    """
    logfile = logging.getLogger('file')
    try:
        if not cache.get("cord19q"):
            logfile = logging.getLogger('file')
            embeddings, db = Models.load(cord19q_path)
            cache.update({'cord19q': {'embeddings': embeddings, 'db': db}})
            logfile.info('finished loaded cord19q data')
            print('finished loaded cord19q data')
        return True
    except Exception as e:
        logfile.exception(e.args)
        return False


def load_model_(model_name, model_path):
    logfile = logging.getLogger('file')
    curr_model.model = None
    # check if model has already been loaded
    if not cache or model_name not in cache:
        if not model_path:
            # we don't have a path to load the model artifacts, log error anr return
            logfile.error('path not found for model: {0}'.format(model_name))
            return False
        else:
            logfile.info('found path for: {0}'.format(model_name))
            try:
                tokenizer = BertTokenizer(
                    **{'vocab_file': os.path.join(model_path, 'vocab.txt'), 'max_len': 512, 'do_lower_case': True})
                tokenizer.unique_added_tokens_encoder.update(set(tokenizer.all_special_tokens))
                model = BertForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=model_path)
                # model.half() not implemented on CPUs!!
                models = []
                num_gpus = 1
                if use_cuda and torch.cuda.is_available():
                    for gpu_id in range(0, num_gpus):
                        with torch.no_grad():
                            models.append(deepcopy(model).cuda(gpu_id))  # separate model for each GPU if multiple GPUs.
                else:
                    models.append(model)
                cache.update({model_name: {'model': models, 'tokenizer': tokenizer}})
                logfile.info('model: {0} successfully loaded'.format(model_name))
            except:
                logfile.exception('Found path but error loading model: {0}'.format(model_name))
                return False
    curr_model.model = model_name
    return True


# The idea here is to run the initialization tasks that need to finish before the flask server can start
# as separate threads so they can execute in parallel. The tasks are:
# task1: load the BERT models
# task2: load the word embeddings, and the articles sqllite database
# task3: parse the metadata.csv file and create a separate dictionary
# task4: query the number of sentences and number of articles from the articles sqllite database and cache this info
# we don't want to query this info every time a request arrives because it takes a non-trivial amount of time to query
def run_init_tasks():
    # use concurrent.futures rather than Threads because that we can get back returned values.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(task3)]
        # If cuda is available, load the large BERT model, otherwise load the base BERT model.
        if use_cuda and torch.cuda.is_available():  # use large bert model if cuda available, otherwise use small
            #futures.append(executor.submit(task1, 'bert_large', bert_large_path))
            futures.append(executor.submit(task1, 'bert_large', bert_large_path))
            print('using large BERT model')
        else:
            futures.append(executor.submit(task1, 'bert_base', bert_small_path))
            print('using small BERT model')

        futures.append(executor.submit(task2))
        futures.append(executor.submit(task4))
        ret = True
        # return False if any task fails
        for future in futures:
            ret = ret & future.result()
        return ret