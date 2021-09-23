import pickle
from os import path

import numpy as np
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report

from utils import read_file, preprocess, stem

stemmer = PorterStemmer()

# Load processed data: raw -> cleaning -> stemming
processed_datafile = 'processed.pkl'
if path.exists(processed_datafile):
    with open(processed_datafile, 'rb') as f:
        data, y = pickle.load(f)
else:
    data, y = read_file('../hatespeech', True)
    data = [preprocess(text) for text in data]
    with open('processed.pkl', 'wb') as f:
        pickle.dump((data, y), f)


keywords = {
    0: ['love', 'people', 'time', 'day', 'life'],
    1: ['free', 'video', 'join', 'check', 'win'],
    2: ['fucked', 'ass', 'bitch', 'bad', 'shit'],
    3: ['hate', 'nigga', 'idiot', 'ass', 'trump']
}



for class_label, words in keywords.items():
    keywords[class_label] = [stem(w) for w in words]

# get count features
count_vectorizer = CountVectorizer(input='content', encoding='ascii',
                                   decode_error='ignore',
                                   strip_accents='ascii',
                                   stop_words='english', min_df=2)
count_weights = count_vectorizer.fit_transform(data)
vocabulary = count_vectorizer.vocabulary_

# get tf idf features
vectorizer = TfidfVectorizer(input='content', encoding='ascii',
                             decode_error='ignore', strip_accents='ascii',
                             stop_words='english', min_df=2,
                             vocabulary=vocabulary)
tfidf_weights = vectorizer.fit_transform(data)

# get tf idf weights for keywords
keyword_indices = {}
for class_label, words in keywords.items():
    keyword_indices[class_label] = [vocabulary[w] for w in words]

class_weights = []
for class_label in range(4):
    indices = keyword_indices[class_label]
    tfidf =  np.array(tfidf_weights[:, indices].sum(axis=1)).flatten()
    weighted_tfidf = np.array(count_weights[:, indices].sum(
        axis=1)).flatten() * tfidf
    class_weights.append(weighted_tfidf)

# assign label based on aggregate
class_weights = np.vstack(class_weights).T
y_pred = []
max_class = np.argmax(class_weights, axis=1)
for i in range(len(data)):
    y_pred.append(max_class[i])

# print classification report
print(classification_report(y, y_pred))
