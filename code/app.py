from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np

app = Flask(__name__)
api = Api(app)

# load the pickled model
with open('./trained_models/SimpleSentimentClassifier.pkl', 'rb') as f:
    model = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    def get(self):
        # use the parser and find user's query
        args = parser.parse_args()
        user_query = args['query']

        # make a prediction on user's query using the model pipeline
        pred = model.predict(np.array([user_query]))
        pred_prob = model.predict_proba(np.array([user_query]))

        # output negative or positive
        if pred == 1:
            pred_text = 'Positive'
            confidence = round(pred_prob[1], 3)
        else:
            pred_text = 'Negative'
            confidence = round(pred_prob[0], 3)

        # create JSON
        output = {'prediction': pred_text, 'confidence': confidence}

        return output