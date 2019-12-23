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

sns.heatmap(df.corr(), cmap='coolwarm', annot=True)  # feedback is our y!
plt.tight_layout()
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

    #return no_stopwords
    return stemmed


# discard irrelevant columns
df_txt = df[['feedback','verified_reviews']]

# learn only from extreme cases - 1 or 5 ratings
#df_txt = df_txt[(df_txt['rating'] == 1) | (df_txt['rating'] == 5)]
df_txt.groupby('feedback').count().iloc[:,0]  # we have class imbalance

# apply the text_process function
# df_txt['verified_reviews'] = df_txt['verified_reviews'].apply(text_process)
# df_txt.head()




from imblearn.over_sampling import SMOTE, SVMSMOTE
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
X = df_txt['verified_reviews']
y = df_txt['feedback']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from imblearn.pipeline import make_pipeline
pipeline = make_pipeline(CountVectorizer(analyzer=text_process),\
                         TfidfTransformer(),\
                         SMOTE(random_state=2431, sampling_strategy='all'),\
                LogisticRegression(max_iter=1000))

# TfidfTransformer(),\
# XGBClassifier(n_estimators=2000, learning_rate=0.05, colsample_bytree=0.7,subsample=0.8, gamma=2)
# SMOTE(random_state=2431, sampling_strategy='all'),\
# RandomForestClassifier(n_estimators=100, criterion='entropy')


pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

# ROC curve
import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, pred, pos_label=1)
roc_auc = metrics.auc(fpr,tpr)

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc}')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# TODO: write mycode that finds the top words in each class (that help to predict)