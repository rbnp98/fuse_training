import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

dataset = pd.read_csv('/home/prajin/Desktop/Sentimnet Analysis/fuse_training-master/ml/data/ISEAR.csv')
dataset.columns = ['Sno', 'emotion','sentence']
dataset = dataset.drop('Sno', axis=1)

def clean_data(review):
    lemmatizer = WordNetLemmatizer()
    eng_stopwords = set(stopwords.words('english'))
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split() 
    review =  [lemmatizer.lemmatize(word) for word in review if word not in eng_stopwords]
    review = ' '.join(review)
    return review

def create_corpus(dataset):
    corpus = []
    for i in range(0,len(dataset)):
        review = dataset['sentence'][i]
        clean_review = clean_data(review)
        corpus.append(clean_review)
    return corpus

dataset_corpus = create_corpus(dataset)


def partition(x):
    if x == 'joy':
        return 1
    if x == 'sadness':
        return 2
    if x == 'anger':
        return 3
    if x =='fear':
        return 4
    if x =='shame':
        return 5
    if x == 'disgust':
        return 6
    if x == 'guilt':
        return 7

def make_data():
    actual_label = dataset.iloc[:,0]
    changed_label = actual_label.map(partition)
    dataset.iloc[:,0] = changed_label
    return dataset, dataset_corpus


