from flask import Flask, request, jsonify

import joblib


app = Flask(__name__)

@app.route('/')
def hello():
	return "Hello World!"


@app.route('/hes')
def hes():
	return "Hes! - Updated again 2"


@app.route('/spam-classifier', methods = ['POST'])
def spam_classifier():
	loaded_model = joblib.load('spam_classifier.pkl')

	text = request.form['text']

	resp = {
		'text': text
	}

	return jsonify(resp)



if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8001, debug=True)