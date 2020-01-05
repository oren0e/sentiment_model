from zipfile import ZipFile
import pandas as pd
from model import SimpleNLPModel

def construct_model():
    model = SimpleNLPModel()

    # extract data
    with ZipFile('./data/amazon-alexa-reviews.zip', 'r') as zip_file:
        zip_file.extractall('../data/')
    df = pd.read_csv('./data/amazon_alexa.tsv', sep='\t')

    model_trained = model.create_model(df)
    model.pickle_classifier(model_trained)

if __name__ == "__main__":
    construct_model()
