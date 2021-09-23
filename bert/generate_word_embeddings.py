import pickle
from os import path
import sys
sys.path.append('../')
import flair
import numpy as np
import torch
from flair.data import Sentence
from flair.embeddings import BertEmbeddings
from nltk import sent_tokenize

from utils import read_file, pad_sequences, clean_doc

if torch.cuda.is_available():
    flair.device = torch.device('cuda:0')
else:
    flair.device = torch.device('cpu')


def get_all_embeddings(data, embedding_model):
    except_counter = 0
    all_embeddings = []
    for index, text in enumerate(data):
        embeddings = []
        if index % 100 == 0:
            print("Finished sentences: " + str(index) + " out of " + str(
                len(data)))
        sentences = sent_tokenize(text)

        for sentence_ind, sent in enumerate(sentences):

            sentence = Sentence(sent, use_tokenizer=True)
            try:
                embedding_model.embed(sentence)
            except Exception as e:
                except_counter += 1
                print("Exception Counter: ", except_counter, sentence_ind,
                      index, e)
                continue
            for token_ind, token in enumerate(sentence):
                word = token.text
                vec = token.embedding.cpu().numpy()
                embeddings.append(vec)
        all_embeddings.append(np.array(embeddings))
    all_embeddings = np.array(all_embeddings)
    with open("../hatespeech/bert_embedding", "wb") as f:
        f.dump(all_embeddings)


if __name__ == "__main__":

    data_dir = '../hatespeech'

    # read cleaned data
    processed_datafile = 'processed.pkl'
    if path.exists(processed_datafile):
        with open(processed_datafile, 'rb') as f:
            processed_data = pickle.load(f)
    else:
        data, y = read_file(data_dir, False)
        data = data
        data = clean_doc(data, True)

        tokenized_data = [s.split() for s in data]
        padded_data = pad_sequences(tokenized_data, 'PAD')
        processed_data = [" ".join(tokens) for tokens in padded_data]
        with open('processed.pkl', 'wb') as f:
            pickle.dump(processed_data, f)

    model = BertEmbeddings('bert-base-uncased')
    get_all_embeddings(processed_data, model)
