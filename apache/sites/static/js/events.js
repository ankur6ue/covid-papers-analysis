var Client = (function (client) {
    let this1 = this;

    $('#submitBtn').click(function () {
        let query = $('#query')[0].value
        client.run(query)
    })
    return client;
}(Client || {}));