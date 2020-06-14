from flask.views import MethodView
from flask import request, jsonify


class HealthCheck(MethodView):
    def get(self):
        return jsonify({'success': True, 'msg': 'OK'})
