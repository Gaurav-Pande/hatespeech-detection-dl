import csv
import itertools
import re
from collections import Counter
from os.path import join

import numpy as np
from nltk import tokenize

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def extract_tfidf_keywords(dataset, vocab, class_type, num_keywords, data,
                        perm):
    sup_data = []
    sup_idx = []
    sup_label = []
    file_name = 'doc_id.txt'
    infile = open(join('../' +dataset + '/', file_name), mode='r',
                  encoding='utf-8')
    text = infile.readlines()
    for i, line in enumerate(text):
        line = line.split('\n')[0]
        class_id, doc_ids = line.split(':')
        assert int(class_id) == i
        seed_idx = doc_ids.split(',')
        seed_idx = [int(idx) for idx in seed_idx]
        sup_idx.append(seed_idx)
        for idx in seed_idx:
            sup_data.append(" ".join(data[idx]))
            sup_label.append(i)

    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk

    tfidf = TfidfVectorizer(norm='l2', sublinear_tf=True, max_df=0.2,
                            stop_words='english')
    sup_x = tfidf.fit_transform(sup_data)
    sup_x = np.asarray(sup_x.todense())

    vocab_dict = tfidf.vocabulary_
    vocab_inv_dict = {v: k for k, v in vocab_dict.items()}

    print("\n### Supervision type: Labeled documents ###")
    print("Extracted keywords for each class: ")
    keywords = []
    cnt = 0
    for i in range(len(sup_idx)):
        class_vec = np.average(sup_x[cnt:cnt + len(sup_idx[i])], axis=0)
        cnt += len(sup_idx[i])
        sort_idx = np.argsort(class_vec)[::-1]
        keyword = []
        if class_type == 'topic':
            j = 0
            k = 0
            while j < num_keywords:
                w = vocab_inv_dict[sort_idx[k]]
                if w in vocab:
                    keyword.append(vocab_inv_dict[sort_idx[k]])
                    j += 1
                k += 1
        elif class_type == 'sentiment':
            j = 0
            k = 0
            while j < num_keywords:
                w = vocab_inv_dict[sort_idx[k]]
                w, t = nltk.pos_tag([w])[0]
                if t.startswith("J") and w in vocab:
                    keyword.append(w)
                    j += 1
                k += 1
        print("Class {}:".format(i))
        print(keyword)
        keywords.append(keyword)

    new_sup_idx = []
    m = {v: k for k, v in enumerate(perm)}
    for seed_idx in sup_idx:
        new_seed_idx = []
        for ele in seed_idx:
            new_seed_idx.append(m[ele])
        new_sup_idx.append(new_seed_idx)
    new_sup_idx = np.asarray(new_sup_idx)

    return keywords, new_sup_idx


def read_file(data_dir, with_evaluation):
    data = []
    target = []
    with open(join(data_dir, 'dataset.csv'), 'rt',
              encoding='utf-8') as csvfile:
        csv.field_size_limit(500 * 1024 * 1024)
        reader = csv.reader(csvfile)
        for row in reader:
            if data_dir == '../hatespeech':
                data.append(row[1])
                target.append(int(row[0]))
    y = None
    if with_evaluation:
        y = np.asarray(target)
        assert len(data) == len(y)
        assert set(range(len(np.unique(y)))) == set(np.unique(y))
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
    # text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = text.lower()
    return text

def clean_general(string):
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def preprocess_doc(data):
    data = [s.strip() for s in data]
    data = [clean_twitter(s) for s in data]
    data = [clean_general(s) for s in data]
    return data


def pad_sequences(sentences, pad_token="<PAD/>", pad_len=None):
    if pad_len is not None:
        sequence_length = pad_len
    else:
        sequence_length = max(len(x) for x in sentences)

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [pad_token] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return word_counts, vocabulary, vocabulary_inv


def build_input_data_cnn(sentences, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in
                 sentences])
    return x


def build_input_matrix(data, vocabulary, max_doc_len):
    x = np.zeros((len(data), max_doc_len), dtype='int32')
    for i, doc in enumerate(data):
        for j, word in enumerate(doc):
            x[i, j] = vocabulary[word]
    return x


def extract_ranking_keywords(dataset, num_keywords, data,
                             perm):
    sup_data = []
    sup_idx = []
    sup_label = []
    file_name = 'doc_id.txt'
    infile = open(join('../' + dataset + '/', file_name), mode='r',
                  encoding='utf-8')
    text = infile.readlines()
    for i, line in enumerate(text):
        line = line.split('\n')[0]
        class_id, doc_ids = line.split(':')
        assert int(class_id) == i
        seed_idx = doc_ids.split(',')
        seed_idx = [int(idx) for idx in seed_idx]
        sup_idx.append(seed_idx)
        for idx in seed_idx:
            sup_data.append(" ".join(data[idx]))
            sup_label.append(i)


    from sklearn.feature_extraction.text import  CountVectorizer

    with open('stopwords.txt', 'r') as f:
        lines = f.readlines()
    stopwords = [w.strip() for w in lines]

    count_vectorizer = CountVectorizer(input='content',
                                       analyzer='word',
                                       strip_accents='ascii',
                                       ngram_range=(1, 1),
                                       stop_words=stopwords,
                                       token_pattern=r'\b[^\d\W]+\b')
    count = count_vectorizer.fit_transform(sup_data)
    features = np.array(count_vectorizer.get_feature_names())
    freq = count.copy()
    count[count > 0] = 1

    print("\n### Supervision type: Labeled documents ###")
    print("Extracted keywords for each class: ")
    keywords = []
    cnt = 0
    rankingdf = pd.DataFrame(columns=['word', 'rel_doc_freq',
                                      'avg_freq', 'idf'])
    rankingdf['word'] = features
    for i in range(len(sup_idx)):
        start = cnt
        end = cnt + len(sup_idx[i])
        cnt += len(sup_idx[i])
        class_docs = count[start: end]
        rel_doc_freq = np.array(class_docs.sum(axis=0) / class_docs.shape[0])[0]
        avg_freq = np.array(freq[start:end].sum(axis=0) /class_docs.shape[0])[0]
        rankingdf['rel_doc_freq'] = rel_doc_freq
        rankingdf['avg_freq'] = avg_freq
        rankingdf['idf'] = np.log(np.array(count.shape[0] / count.sum(axis=0))[0])

        scaler = MinMaxScaler()
        scaler.fit(rankingdf[['rel_doc_freq', 'idf', 'avg_freq']])
        rankingdf[['rel_doc_freq', 'idf', 'avg_freq']] = scaler.transform(rankingdf[['rel_doc_freq', 'idf', 'avg_freq']])
        rankingdf['comb'] = np.cbrt(rankingdf['rel_doc_freq'] * rankingdf['idf'] * rankingdf['avg_freq'])
        keyword = rankingdf.sort_values(by=['comb'], ascending=False).head(
            num_keywords)['word'].tolist()
        keywords.append(keyword)


    new_sup_idx = []
    m = {v: k for k, v in enumerate(perm)}
    for seed_idx in sup_idx:
        new_seed_idx = []
        for ele in seed_idx:
            new_seed_idx.append(m[ele])
        new_sup_idx.append(new_seed_idx)
    new_sup_idx = np.asarray(new_sup_idx)

    return keywords, new_sup_idx


def load_keywords(dataset='hatespeech'):
    file_name = 'keywords.txt'
    print("\n### Supervision type: Class-related Keywords ###")
    print("Keywords for each class: ")
    infile = open(join('../' + dataset + '/', file_name), mode='r',
                  encoding='utf-8')
    text = infile.readlines()

    keywords = []
    for i, line in enumerate(text):
        line = line.split('\n')[0]
        class_id, contents = line.split(':')
        assert int(class_id) == i
        keyword = contents.split(',')
        print("Supervision content of class {}:".format(i))
        print(keyword)
        keywords.append(keyword)
    return keywords




def get_train_inputs(data, y, vocabulary, vocabulary_inv, sup_source,
                     num_keywords=5, with_evaluation=True, dataset_name =
             'hatespeech', keyword_method='tfidf'):
    x = build_input_matrix(data, vocabulary, len(data[0]))

    sz = len(x)
    np.random.seed(1234)
    perm = np.random.permutation(sz)
    x = x[perm]

    if with_evaluation:
        print("Number of classes: {}".format(len(np.unique(y))))
        print("Number of documents in each class:")
        for i in range(len(np.unique(y))):
            print("Class {}: {}".format(i, len(np.where(y == i)[0])))
        y = y[perm]

    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    if sup_source == 'labels' or sup_source == 'keywords':
        keywords = load_keywords(dataset_name)
        return x, y, keywords, perm
    elif sup_source == 'docs':
        class_type = 'topic'
        if keyword_method == 'tfidf':
            keywords, sup_idx = extract_tfidf_keywords(dataset_name, vocabulary,
                                                       class_type,
                                                       num_keywords, data,
                                                       perm)
        else:
            keywords, sup_idx = extract_ranking_keywords(dataset_name, vocabulary,
                                                         class_type,
                                                         num_keywords, data,
                                                         perm)
        return x, y, keywords, sup_idx, perm


def load_dataset(dataset_name, sup_source, with_evaluation=True,
                 keyword_method='tfidf'):
    return get_train_inputs(dataset_name, sup_source,
                            with_evaluation=with_evaluation,
                            keyword_method=keyword_method)

def print_len_stats(data):
    tmp_list = [len(doc) for doc in data]
    len_max = max(tmp_list)
    len_avg = np.average(tmp_list)
    len_std = np.std(tmp_list)

    print("\n### Dataset statistics: ###")
    print('Document max length: {} (words)'.format(len_max))
    print('Document average length: {} (words)'.format(len_avg))
    print('Document length std: {} (words)'.format(len_std))
    return len_max, len_avg, len_std

def load_dataset_v2(dataset_name, with_evaluation):
    data_path = '../' + dataset_name
    data, y = read_file(data_path, with_evaluation)

    data = preprocess_doc(data)
    data = [s.split(" ") for s in data]

    len_max, len_avg, len_std = print_len_stats(data)

    padded_data = pad_sequences(data)
    word_counts, vocabulary, vocabulary_inv = build_vocab(padded_data)
    return padded_data, y, word_counts, vocabulary, vocabulary_inv, len_max, len_avg, len_std
