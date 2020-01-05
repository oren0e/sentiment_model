# Demo Sentiment Model on AWS 
## Overview
This is a very simple sentiment NLP model which tries to classify a review as "positive" or "negative" and returns
the confidence of its prediction as well. The model's performance is extremely poor and is heavily biased towards the "positive"
prediction case. This is because the goal of this repo is rather technical: To demonstrate a model that was developed with
Python Flask and deployed to AWS running on Docker image.    

The data for training this model came from the [Amazon Alexa Reviews](https://www.kaggle.com/sid321axn/amazon-alexa-reviews) dataset.

## Usage
`curl -X GET http://sentiment-api-demo.us-east-2.elasticbeanstalk.com/ -d query='hate bad wrong'`
`{"prediction": "Negative", "confidence": 0.521}`

`curl -X GET http://sentiment-api-demo.us-east-2.elasticbeanstalk.com/ -d query='cool awesome great'`
`{"prediction": "Positive", "confidence": 0.972}`
