from flask import Flask, request, jsonify

from TextClassifier import TextClassifier

import torch
import torch.nn as nn

import joblib


app = Flask(__name__)

@app.route('/')
def hello():
	return "Hello World!"


@app.route('/hes')
def hes():
	return "Hes! - Updated again 4"


@app.route('/spam-classifier', methods = ['POST'])
def spam_classifier():
	loaded_model = joblib.load('spam_classifier.pkl')

	text = request.form['text']
	result = loaded_model.predict([text])

	resp = {
		'result': result.tolist()
	}

	return jsonify(resp)

@app.route('/spam-classifier-dl', methods = ['POST'])
def spam_classifier_dl():
	loaded_model = torch.load('spam_classifier_model_dl_2_1.pth') # it takes the loaded dictionary, not the path file itself
	loaded_model.eval()

	spam_label = {0: "Ham", 1: "Spam"}

	text = [request.form['text']]

	vectorizer = joblib.load('vectorizer_2_1.pkl')

	text = torch.tensor(vectorizer.transform(text).toarray(), dtype=torch.float32)
	output = loaded_model(text).squeeze()

	label_predict = 0 if output < 0.5 else 1

	resp = {
		'result': spam_label[label_predict],
	}

	return jsonify(resp)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8001, debug=True)
