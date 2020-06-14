
def log_request_info(req):
    req_info = {}
    req_info['access_route'] = req.access_route
    req_info['base_url'] = req.base_url
    req_info['endpoint'] = req.endpoint
    req_info['host_url'] = req.host_url
    req_info['http_host'] = req.environ['HTTP_HOST']
    req_info['request_uri'] = req.environ['REQUEST_URI']
    req_info['remote_addr'] = req.environ['REMOTE_ADDR']
    req_info['remote_port'] = req.environ['REMOTE_PORT']
    req_info['server_name'] = req.environ['SERVER_NAME']
    req_info['server_port'] = req.environ['SERVER_PORT']
    req_info['remote_addr'] = req.environ['REMOTE_ADDR']
    req_info['http_origin'] = req.environ['HTTP_ORIGIN']
    req_info['http_referer'] = req.environ['HTTP_REFERER']
    req_info['http_x_forwarded_for'] = req.environ['HTTP_X_FORWARDED_FOR']
    req_info['http_x_forwarded_host'] = req.environ['HTTP_X_FORWARDED_HOST']
    req_info['http_x_forwarded_server'] = req.environ['HTTP_X_FORWARDED_SERVER']

    print('HTTP Header Info:')
    for k, v in req_info.items():
        print('{0}: {1}'.format(k, v))


