from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

pd.set_option('expand_frame_repr', False)  # To view all the variables in the console

# extract data
with ZipFile('./data/amazon-alexa-reviews.zip', 'r') as zip_file:
    zip_file.extractall('./data/')

df = pd.read_csv('./data/amazon_alexa.tsv', sep='\t')
df.head()
df.info()
df.describe()

# EDA #

# how many for each rating
df.groupby('rating').count().iloc[:,1]

# how long are reviews
df['review_length'] = df['verified_reviews'].apply(len)
df.groupby('rating').median().loc[:,'review_length']

df.hist(column='review_length', by='rating', bins=30, edgecolor="black")
plt.show()

## one case for example ##
case = df['verified_reviews'][21]
# 1. remove punctuation
case_no_punc = ''.join([word for word in case if word not in string.punctuation]).lower()
# 2. remove stop words
case_no_stopwords = ' '.join([word for word in case_no_punc.split(' ') if word not in stopwords.words('english')])
# 3. stem words
case_stemmed = ' '.join([PorterStemmer().stem(word) for word in case_no_stopwords.split(' ')])

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
df_txt.groupby('rating').count().iloc[:,0]  # we have class imbalance

from imblearn.over_sampling import SMOTE

#TODO: 1. use SMOTE
#      2. use train test split
#      3. run naivebayes model
#      4. write a class for model
#      5. build API
