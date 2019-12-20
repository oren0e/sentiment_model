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
        pred = model.predict()