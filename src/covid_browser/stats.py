from flask.views import MethodView
from flask import request, jsonify
from .globals import cache


class Stats(MethodView):
    def __init__(self, path):
        self.path = path

    def get(self):
        return jsonify({'success': True, 'num_sentences': cache.get('num_all_sentences')[0],
                        'num_articles': cache.get('num_all_articles')[0]})
