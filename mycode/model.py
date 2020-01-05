from zipfile import ZipFile
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline

# text process function
class SimpleNLPModel():

    @staticmethod
    def text_process(text: str) -> str:
        """
        1. Remove punctuation
        2. Remove stopwords
        3. Stem words using PorterStemmer()
        """
        no_punc = ''.join([word for word in text if word not in string.punctuation]).lower()
        no_stopwords = ' '.join([word for word in no_punc.split(' ') if word not in stopwords.words('english')])
        stemmed = ' '.join([PorterStemmer().stem(word) for word in no_stopwords.split(' ')])
        return stemmed

    def create_model(self, df):
        # discard irrelevant columns
        df_txt = df[['feedback','verified_reviews']]

        # train test split
        X = df_txt['verified_reviews']
        y = df_txt['feedback']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

        # make a modeling pipeline
        pipeline = make_pipeline(CountVectorizer(analyzer=self.text_process),\
                                 TfidfTransformer(),\
                                 SMOTE(random_state=2431, sampling_strategy='all'),\
                                 LogisticRegression(max_iter=1000))
        # train the model
        pipeline.fit(X_train, y_train)

        return pipeline

    @staticmethod
    def pickle_classifier(clf, path='./trained_models/SimpleSentimentClassifier.pkl'):
        # pickle the classifier
        with open(path, 'wb') as f:
            pickle.dump(clf, f)
            print('model pickled!')
