import torch
import torch.nn as nn

from sklearn.feature_extraction.text import TfidfVectorizer

import joblib


class TextClassifier(nn.Module):
  def __init__(self, in_features):
    super(TextClassifier, self).__init__()

    self.layer_1 = nn.Linear(in_features, 128)
    self.layer_2 = nn.Linear(128, 64)
    self.layer_3 = nn.Linear(64, 1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    self.init_weights()

  def init_weights(self):
      initrange = 0.5

      self.layer_1.weight.data.uniform_(-initrange, initrange)
      self.layer_1.bias.data.zero_()

      self.layer_2.weight.data.uniform_(-initrange, initrange)
      self.layer_2.bias.data.zero_()

      self.layer_3.weight.data.uniform_(-initrange, initrange)
      self.layer_3.bias.data.zero_()

  def forward(self, x):

    z = self.layer_1(x)
    z = self.layer_2(z)
    z = self.layer_3(z)

    z = self.relu(self.layer_1(x))
    z = self.relu(self.layer_2(z))
    z = self.layer_3(z)

    z = self.sigmoid(z)  # if use BCELoss as Loss Function, enable this

    return z

  # def predict(self, text, model):
  #   with torch.no_grad():
  #     vectorizer = joblib.load('vectorizer.pkl')

  #     text = torch.tensor(vectorizer.transform(text).toarray(), dtype=torch.float32)
  #     output = model(text).squeeze()

  #     label_predict = 0 if output < 0.5 else 1

  #     return (label_predict, output)