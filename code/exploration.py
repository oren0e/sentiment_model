from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

