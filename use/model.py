import pickle
from os import path

import numpy as np
from nltk.stem.porter import *
from utils import read_file, preprocess, stem
import tensorflow as tf
import tensorflow_hub as hub 

def get_use_embeddings():
    processed_datafile = 'processed.pkl'
    if path.exists(processed_datafile):
        with open(processed_datafile, 'rb') as f:
            data, y = pickle.load(f)
    else:
        data, y = read_file('../hatespeech', True)
        data = [preprocess(text) for text in data]
        with open('processed.pkl', 'wb') as f:
            pickle.dump((data, y), f)

    vocab = ['<PAD/>']
    vocab_file = 'vocab.npy'
    if path.exists(vocab_file):
        vocab = np.load(vocab_file, allow_pickle='TRUE')
    else:
        for sentence in data:
            words = sentence.split()
            for word in words:
                if word not in vocab:
                    vocab.append(word)
        np.save(vocab_file, vocab)

    def embed(input):
        return model(input)

    embed_file = '../hatespeech/use_embeddings.npy'
    embeddings = {}

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)

    message_embeddings = embed(vocab)

    for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        embeddings[vocab[i]] = (message_embedding)
    
    np.save(embed_file, embeddings)
    return embeddings