from flask import Flask
from flask_restful import reqparse, Api, Resource
import pickle
import numpy as np



app = Flask(__name__)
api = Api(app)

# load the pickled model
with open('../trained_models/SimpleSentimentClassifier.pkl', 'rb') as f:
    model1 = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


# Define a resource (what to do when a URL is accessed)
class PredictSentiment(Resource):
    def get(self):
        # use the parser and find user's query
        args = parser.parse_args()
        user_query = args['query']

        # make a prediction on user's query using the model pipeline
        pred = model1.predict(np.array([user_query]))
        pred_prob = model1.predict_proba(np.array([user_query]))

        # output negative or positive
        if pred == 1:
            pred_text = 'Positive'
            confidence = round(pred_prob[0][1], 3)
        else:
            pred_text = 'Negative'
            confidence = round(pred_prob[0][0], 3)

        # create JSON
        output = {'prediction': pred_text, 'confidence': confidence}

        return output

# Route the resource to the URL
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
