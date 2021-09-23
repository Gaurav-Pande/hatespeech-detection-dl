import csv
from os.path import join

import numpy as np
import spacy
from nltk.stem.porter import *

# nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()


def read_file(data_dir, with_evaluation):
    data = []
    target = []
    with open(join(data_dir, 'dataset.csv'), 'rt',
              encoding='utf-8') as csvfile:
        csv.field_size_limit(500 * 1024 * 1024)
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row[1])
            target.append(int(row[0]))
    if with_evaluation:
        y = np.asarray(target)
        assert len(data) == len(y)
        assert set(range(len(np.unique(y)))) == set(np.unique(y))
    else:
        y = None
    return data, y


def clean_twitter(text):
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r'\w+@\w+\.\w+', ' ', text)  # remove emails
    text = re.sub(r'(https{0,1}\:\/\/.+?(\s|$))|(www\..+?(\s|$))|('
                  r'\b\w+\.twitter\.com.+?(\s|$))', ' ',
                  text)  # remove urls
    text = re.sub(r'(@[A-Za-z0-9_]+:?(\s|$))', ' ', text)  # remove mentions
    text = re.sub(r'\b(RT|rt)\b', ' ', text)  # remove retweets
    text = re.sub(r'(&#\d+;)+', ' ', text)  # remove retweets
    text = re.sub(r'&\w+;(\w)?', ' ', text)
    text = re.sub(r'(#[A-Za-z0-9_]+)', ' ', text)  # remove hashtags
    text = re.sub(r'(\.){2,}', '.', text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[^A-Za-z0-9(),.!?_\"\'\`]", " ", text)
    text = text.lower()
    return text

def separate_punctuation(string):
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\"", " \" ", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\$", " $ ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    return string.strip().lower()


def clean_doc(data, is_tweet=False):
    data = [s.strip() for s in data]
    if is_tweet:
        data = [clean_twitter(s) for s in data]
    data = [separate_punctuation(s) for s in data]
    return data


def stem(sentence):
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    sentence = ' '.join(stemmed_tokens)
    return sentence


def preprocess(text):
    text = clean_twitter(text)
    text = stem(text)
    return text

def pad_sequences(sentences, padding_word="<PAD/>", pad_len=None):
    if pad_len is not None:
        sequence_length = pad_len
    else:
        sequence_length = max(len(x) for x in sentences)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        print(sentence)
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

