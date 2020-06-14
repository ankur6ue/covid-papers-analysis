var Client = (function () {
    //Parameters
    let client = {}
    let document_ta = $("#document")[0] // ta is text area
    const s = document.getElementById('covid-19');
    let api_server = s.getAttribute('serv_addr')
    let log = $("#log")[0]
    log.value = ""

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

        // Disable submit button so users can't send repeated requests
        $("#submitBtn").prop("disabled", true)
        let xhr = new XMLHttpRequest()
        xhr.open('POST', 'cord19q_lookup/' + query)
        // Only accept JSON styles response back
        xhr.setRequestHeader('Accept', 'application/JSON');
        let send_t = new Date().getTime()
        xhr.onload = function () {
            $("#submitBtn").prop("disabled", false)
            let results = JSON.parse(this.response)
            if (this.status === 200) {
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
                            let score = vals[0][0][0].toFixed(2)
                            let answer = vals[0][0][1]
                            let title = vals[1]
                            let date = vals[2]
                            let url = vals[3]
                            if (url)
                                table_data.push({answer: answer, score: score,
                                    title: "<a href=" + url + ">" + title + "</a>", date: date})
                            else
                                table_data.push({answer: answer, score: score,
                                    title: title, date: date})
                        }
                    }
                    // don't know why, but both appear to be needed
                    $table.bootstrapTable('load', table_data)
                    $table.bootstrapTable({data: table_data})

                    // Now deal with the excerpts (answer spans)
                    // First make the excerpt div visible
                    $("#Excerpts").addClass("visible")
                    $("#Excerpts").removeClass("invisible")
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
                            html: '<h6>' + '<b>' + "excerpt " + num + '</b>' + '</h6>' + '<p>' + text_ + '</p>'
                        }).appendTo($(".results"))
                    })
                } else {
                    log.value += '\n' + results.msg
                    client.update_scroll(log)
                }
            } else {
                log.value += '\n' + results.msg
                client.update_scroll(log)
            }

        }
        xhr.send(); // appears to automatically strip alpha-numeric characters..
    }

    function get_stats() {
        let xhr = new XMLHttpRequest()
        xhr.open('GET', 'stats')
        xhr.setRequestHeader('Accept', 'application/JSON');
        xhr.onload = function () {
            if (this.status === 200) {
                let result = JSON.parse(this.response)
                if (result.success) {
                    num_sentences = result.num_sentences
                    num_articles = result.num_articles
                     $('<p>' +
                        "This demo uses a combination of Natural Language Processing and AI techniques to find best \
                         matching sentences and excerpts for a \
                        user query from a dataset of " + num_articles + ' research papers containing ' + num_sentences +
                        ' sentences about Covid-19 and other infectious \
                        diseases. Try searching for "how is covid-19 transmitted?" or "what are the major risk factors for \
                        covid-19?"' +
                        '</p>'
                        ).appendTo($(".intro-text"))
                }
            }
        }
        xhr.send();

    }
    // populate_models_dropdown(client.models)
    // populate_titles_dropdown()
    get_stats()
    return client
}());