from zipfile import ZipFile
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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

def create_model():
    # extract data
    with ZipFile('./data/amazon-alexa-reviews.zip', 'r') as zip_file:
        zip_file.extractall('../data/')

    df = pd.read_csv('./data/amazon_alexa.tsv', sep='\t')

    # discard irrelevant columns
    df_txt = df[['feedback','verified_reviews']]

    # train test split
    from sklearn.model_selection import train_test_split
    X = df_txt['verified_reviews']
    y = df_txt['feedback']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

    # make a modeling pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from imblearn.over_sampling import SMOTE
    from sklearn.linear_model import LogisticRegression
    from imblearn.pipeline import make_pipeline
    pipeline = make_pipeline(CountVectorizer(analyzer=text_process),\
                             TfidfTransformer(),\
                             SMOTE(random_state=2431, sampling_strategy='all'),\
                             LogisticRegression(max_iter=1000))
    # train the model
    pipeline.fit(X_train, y_train)

    # pickle the classifier
    import pickle
    with open('./trained_models/SimpleSentimentClassifier.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
        print('model pickled!')

if __name__ == "__main__":
    create_model()