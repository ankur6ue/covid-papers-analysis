var Client = (function (client) {
    let this1 = this;
    let log = $("#log")[0]
    const regex = /^[a-zA-Z0-9 ?-]+$/
    $('#submitBtn').click(function () {
        let query = $('#query')[0].value
        if (query.length < 3 || query.length > 200) {
            log.value += '\n' + 'Query must be non-zero length and contain fewer than 200 alphanumeric characters'
            return
        }
        if (query.match(regex)) {
            client.run(query)
        } else {
            log.value += '\n' + 'Query must not contain non-alphanumeric characters (except ? and -)'
            return
        }
    })
    return client;
}(Client || {}));