from flask import Flask, request, jsonify
from nlp import stemming, textrank
app = Flask(__name__)


@app.route('/process_text', methods=['POST'])
def process_text():
    input_json = request.get_json()
    if not bool(input_json):
        return {'error': 'no text provided'}, 400
    processed_text = stemming.preprocess(input_json['text'])
    return jsonify(processed_text)


@app.route("/sentence_similarity_score", methods=['POST'])
def sentence_similarity_score():
    input_json = request.get_json()
    if input_json['sent1'] is None or input_json['sent1'] is None:
        return {'error': 'no text provided'}, 400
    score = textrank.sentence_similarity(input_json['sent1'],
                                         input_json['sent2'])
    return str(score)


@app.route("/textrank", methods=['POST'])
def textrank():
    input_json = request.get_json()
    if input_json['sentences'] is None:
        return {'error': 'no text provided'}, 400
    return textrank.textrank(input_json['sentences'])


if __name__ == '__main__':
    app.run()
