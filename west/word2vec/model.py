import itertools
import os
import pickle
import re
from collections import Counter
from os import path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import word2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from  utils import read_file, clean_twitter
nltk_stopwords = stopwords.words('english')

def remove_punctuation(text):
    text = re.sub(r"[^A-Za-z`]", " ", text)
    return text


def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return word_counts, vocabulary, vocabulary_inv


def get_embeddings(inp_data, vocabulary_inv, size_features=100,
                   mode='skipgram',
                   min_word_count=2,
                   context=5):
    model_name = "embedding"
    model_name = os.path.join(model_name)
    if os.path.exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print("Loading existing Word2Vec model {}...".format(model_name))
    else:
        num_workers = 15  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in inp_data]
        if mode == 'skipgram':
            sg = 1
            print('Model: skip-gram')
        elif mode == 'cbow':
            sg = 0
            print('Model: CBOW')
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            sg=sg,
                                            size=size_features,
                                            min_count=min_word_count,
                                            window=context,
                                            sample=downsampling)

        embedding_model.init_sims(replace=True)
        print("Saving Word2Vec model {}".format(model_name))
        embedding_model.save(model_name)
    embedding_weights = np.zeros((len(vocabulary_inv), size_features))

    for i in range(len(vocabulary_inv)):
        word = vocabulary_inv[i]
        if word in embedding_model:
            embedding_weights[i] = embedding_model[word]
        else:
            embedding_weights[i] = np.random.uniform(-0.25, 0.25,
                                                     embedding_model.vector_size)
    return embedding_weights


# Define keywords
keywords = {
    0: ['love', 'people', 'time', 'day', 'life'],
    1: ['free', 'video', 'join', 'check', 'win'],
    2: ['fucked', 'ass', 'bitch', 'bad', 'shit'],
    3: ['hate', 'nigga', 'idiot', 'ass', 'trump']
}


# Load processed data: raw -> cleaning -> stemming
processed_datafile = 'processed.pkl'
if path.exists(processed_datafile):
    with open(processed_datafile, 'rb') as f:
        data, y = pickle.load(f)
else:
    data, y = read_file('../hatespeech', True)
    data = [clean_twitter(text) for text in data]
    data = [remove_punctuation(text) for text in data]
    tokenized = [word_tokenize(text) for text in data]
    data = tokenized
    with open('processed.pkl', 'wb') as f:
        pickle.dump((data, y), f)

# for doc in data:
#     assert(len(doc) > 0)
word_counts, vocabulary, vocabulary_inv = build_vocab(data)

inp_data = [[vocabulary[word] for word in text] for text in data]

embedding_weights = get_embeddings(inp_data, vocabulary_inv)


def get_label_embeddings():
    label_embeddings = []
    for class_label in range(4):
        embedding_list = []
        for word in keywords[class_label]:
            embedding_list.append(embedding_weights[vocabulary[word]])
        label_embeddings.append(np.array(embedding_list).sum(axis=0) / 5)
    return label_embeddings


def get_doc_embeddings():
    doc_weights = []
    for doc in inp_data:
        if len(doc) == 0:
            doc_weights.append(np.random.uniform(-0.25, 0.25, 100))
        else:
            doc_weights.append(embedding_weights[doc].sum(axis=0) / len(doc))
    return doc_weights


label_embeddings = get_label_embeddings()
doc_embeddings = get_doc_embeddings()

similarities = cosine_similarity(doc_embeddings, label_embeddings)
y_pred = np.argmax(similarities, axis=1)

# print classification report
print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
print(cm)
labels = ['normal', 'spam', 'abusive', 'hateful']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
