var Client = (function () {
    //Parameters
    let client = {}
    let document_ta = $("#document")[0] // ta is text area
    let host_type = 'localhost'
    let log = $("#log")[0]
    log.value = ""
    if (host_type == 'localhost') {
        api_server = "http://127.0.0.1:5000" // must be just like this. using 0.0.0.0 for the IP doesn't work!

    } else {
        api_server = "https://telesens.co/face_det"
    }
    client.models = [
        {
            "name": "bert-large-uncased-whole-word-masking-finetuned-squad",
            "short_name": "bert_large",
            "path": ""
        },
        {
            "name": 'bert-base-uncased_finetuned_squad',
            "short_name": "bert_base",
            "path": '/home/ankur/dev/apps/ML/transformers/models/'
        }
    ];

    client.update_scroll = function (log) {
        log.scrollTop = log.scrollHeight
    }
    // load text
    client.run = function (query) {
        console.log('in run')
        let xhr = new XMLHttpRequest()
        xhr.open('POST', api_server + '/cord19q_lookup/' + query)
        let send_t = new Date().getTime()
        xhr.onload = function () {
            if (this.status === 200) {

                let results = JSON.parse(this.response)
                if (results.success) {
                    cordq_answers = results.cordq_answers
                    let recv_t = new Date().getTime()
                    log.value += '\n' + 'query processing time (ms): ' + (recv_t - send_t).toFixed(0)
                    client.update_scroll(log)
                    let $table = $('#table')
                    let table_data = []
                    for (let key in cordq_answers) {
                        // check if the property/key is defined in the object itself, not in parent
                        if (cordq_answers.hasOwnProperty(key)) {
                            let vals = cordq_answers[key]
                            let score = vals[0][0][0]
                            let answer = vals[0][0][1]
                            let title = vals[1]
                            let date = vals[2]
                            table_data.push({id: key, answer: answer, score: score, title: title, date: date})
                        }
                    }
                    // don't know why, but both appear to be needed
                    $table.bootstrapTable('load', table_data)
                    $table.bootstrapTable({data: table_data})
                    bert_answers = results.bert_answers
                    $(".resultItem").remove();
                    bert_answers.forEach((answer_, index) => {
                        let text_ = answer_.context
                        let start_span = answer_.start
                        let end_span = answer_.end
                        let answer = answer_.answer
                        let num = Number(index) + 1
                        text_ = text_.replace(answer, "<span style=\"background-color: #FFFF00\">" + answer + "</span>")
                        $('<p>', {
                            class: 'resultItem',
                            html: '<h6>' + '<b>' + "result " + num + '</b>' + '</h6>' + '<p>' + text_ + '</p>'
                        }).appendTo($(".results"))
                    })
                } else {
                    log.value += '\n' + 'error processing query. Check if you have selected a model and document'
                    client.update_scroll(log)
                }
            }
        }
        xhr.send();
    }

    function load_model(name) {
        console.log('loading model..')
        let xhr = new XMLHttpRequest()
        xhr.open('POST', api_server + '/load_model/' + name)
        let send_t = new Date().getTime()
        xhr.onload = function () {
            if (this.status === 200) {
                let recv_t = new Date().getTime()
                let result = JSON.parse(this.response)
                if (result.success) {
                    client.active_model_name = name
                    log.value += '\n' + 'active model: ' + name + '\n' + 'load time (ms): ' + (recv_t - send_t).toFixed(0)
                    client.update_scroll(log)
                } else {
                    log.value += '\n' + 'error loading model'
                    client.update_scroll(log)
                }
            }
        }
        xhr.send();
    }

    function populate_models_dropdown(models) {
        let dropdown = $(".dropdown-menu.models");
        $(".dropdown-menu.models").empty();
        let _this = this
        for (let i = 0; i < models.length; i++) {
            list_item = "<li class='list-item ModelsDropDownListItem' data-name='" + models[i].name +
                "' data-path=" + models[i].path + ">"
                + "<a role='menuitem'  href='#'>" + models[i].short_name + "</a>" + "</li>"
            dropdown.append(list_item);
        }
        $('.ModelsDropDownListItem').click(function (e) {
            let target = e.currentTarget;
            let name = target.getAttribute("data-name")
            let path = target.getAttribute("data-path")
            load_model(name)
            // $('#activeModel')[0].value = name;
        });
    }

    function load_abstract(id) {
        let xhr = new XMLHttpRequest()
        xhr.open('POST', api_server + '/get_abstract/' + id)
        xhr.onload = function () {
            if (this.status === 200) {
                let result = JSON.parse(this.response)
                if (result.success) {
                    document_ta.value = result.context
                }
            }
        }
        xhr.send();
    }

    function populate_titles_dropdown() {
        let dropdown = $(".dropdown-menu.scrollable-menu.titles");
        dropdown.empty();
        let _this = this
        let xhr = new XMLHttpRequest()
        xhr.open('POST', api_server + '/get_titles/')
        xhr.onload = function () {
            if (this.status === 200) {
                let result = JSON.parse(this.response)
                if (result.success) {
                    titles = result.titles
                    titles.forEach((title_, index) => {

                        list_item = "<li class='dropdown-item titlesDropDownListItem' data-name=" + title_[0] +
                            " data-id=" + title_[0] +
                            " data-toggle=tooltip data-placement=right title=" + "\"" + title_[1] + "\"" + ">"
                            + "<a role='menuitem'  href='#'>" + title_[1] + "</a>" + "</li>"

                        dropdown.append(list_item)
                        separator = "<div class='dropdown-divider'></div>"
                        dropdown.append(separator)

                    })

                    $('.titlesDropDownListItem').click(function (e) {
                        let target = e.currentTarget;
                        let doc_id = target.getAttribute("data-id")
                        load_abstract(doc_id)
                    });

                }
            }
        }
        xhr.send();

    }

    // populate_models_dropdown(client.models)
    // populate_titles_dropdown()
    return client
}());