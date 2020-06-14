from flask import render_template
from flask import Flask
from flask import jsonify
from flask_cors import CORS
import torch
import os
from .errors import BadQuery, ServerError
import logging
from .globals import base_path
from .modelapi import ModelAPI
from .healthchck import HealthCheck
from .stats import Stats
from .init_tasks import run_init_tasks
torch.set_grad_enabled(False)
application = Flask(__name__)
# this keeps Flask from reordering the sort order of dictionary keys
# during jsonfiy. Otherwise if you have a dict ordered by scores, Flask will serialize JSON objects in a way that the
# keys are ordered. See https://github.com/pallets/flask/issues/974
application.config["JSON_SORT_KEYS"] = False
CORS(application)


# Error handlers
@application.errorhandler(BadQuery)
def handle_bad_query(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@application.errorhandler(ServerError)
def handle_server_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


# set up logging to be used else where in the system
logging.basicConfig(filename= os.path.join(base_path, 'logs/log.txt'), level=logging.INFO)
logfile = logging.getLogger('file')


# if PORT passed as env var (eg from docker run), then use that, else use 5001
port = 5001 if not os.environ.get('PORT') else os.environ.get('PORT')
print('flask running on port: {0}'.format(port))

# same thing with SERVER ADDRESS
serv_addr = os.environ.get('SERV_ADDR') if os.environ.get('SERV_ADDR') else 'http://127.0.0.1:5000'
print('server address: {0}'.format(serv_addr))
logfile.info('server address: {0}'.format(serv_addr))


# set up views
# ModelAPI is where the query look-up is implemented
api_impl = ModelAPI.as_view('model_api')
application.add_url_rule('/cord19q_lookup/<query>', view_func=api_impl, methods=['POST'])

healthchck = HealthCheck.as_view('health_check_api')
application.add_url_rule('/healthcheck', view_func=healthchck, methods=['GET'])

# implements the endpoint that returns stats such as number of sentences
stats = Stats.as_view('stats', base_path)
application.add_url_rule('/stats', view_func=stats, methods=['GET'])


def run():
    print("CUDA availability: {0}".format(torch.cuda.is_available()))
    # will block until all init tasks are complete
    if not run_init_tasks():
        logfile.error('init tasks failed, can not start application')
        return -1
    application.run(debug=False, host='0.0.0.0', port=port)


if __name__ == '__main__':
    run()
