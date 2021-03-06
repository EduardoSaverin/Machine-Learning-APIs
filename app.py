from flask import Flask, request, jsonify
from nlp import stemming, textrank

app = Flask(__name__)
from nlp.keyword import KeyWordExtraction
from nlp.textsummary import TextSummary


@app.route('/process_text', methods=['POST'])
def process_text():
    input_json = request.get_json()
    if not bool(input_json):
        return {'error': 'no text provided'}, 400
    processed_text = stemming.preprocess(input_json['text'], input_json['cleanup'])
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
def generate_summary():
    input_json = request.get_json()
    if input_json['paragraph'] is None:
        return {'error': 'no text provided'}, 400
    return ''.join(textrank.textrank(input_json['paragraph']))


@app.route("/keywords", methods=['POST'])
def find_keywords():
    input_json = request.get_json()
    if input_json['paragraph'] is None:
        return {'error': 'no text provided'}, 400
    keyword = KeyWordExtraction()
    keyword.analyze(input_json['paragraph'])
    return ','.join(keyword.get_keywords())


@app.route("/textsummary", methods=['POST'])
def text_summary():
    input_json = request.get_json()
    if input_json['paragraph'] is None:
        return {'error': 'no text provided'}, 400
    textsummary = TextSummary(input_json['paragraph'], input_json['limit'])
    summary = textsummary.find_summary()
    return summary


if __name__ == '__main__':
    app.run()
