from zipfile import ZipFile
import pandas as pd
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# extract data
with ZipFile('./data/amazon-alexa-reviews.zip', 'r') as zip_file:
    zip_file.extractall('./data/')

df = pd.read_csv('./data/amazon_alexa.tsv', sep='\t')

# text process function
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

# discard irrelevant columns
df_txt = df[['rating','verified_reviews']]

# learn only from extreme cases - 1 or 5 ratings
df_txt = df_txt[(df_txt['rating'] == 1) | (df_txt['rating'] == 5)]

# train test split
from sklearn.model_selection import train_test_split
X = df_txt['verified_reviews']
y = df_txt['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

# make a modeling pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from xgboost import XGBClassifier
from imblearn.pipeline import make_pipeline
pipeline = make_pipeline(CountVectorizer(analyzer=text_process),\
                         TfidfTransformer(),\
                         XGBClassifier(n_estimators=2000, learning_rate=0.05, colsample_bytree=0.7,subsample=0.8, gamma=2))
# train the model
pipeline.fit(X_train, y_train)

# pickle the classifier
import pickle
with open('./trained_models/SimpleSentimentClassifier.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
    print('model pickled!')

if __name__ == "__main__":
    model()