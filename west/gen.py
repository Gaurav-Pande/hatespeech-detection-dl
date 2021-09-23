import numpy as np
import os
np.random.seed(1234)
from spherecluster import SphericalKMeans, VonMisesFisherMixture, sample_vMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def seed_expansion(word_sup_array, prob_sup_array, sz, write_path, vocabulary_inv, embedding_mat):
    # print("input to seed word")
    # print("word_sup_array", word_sup_array)
    # print('prob_sup_array', prob_sup_array)
    # print('sz',sz)
    # print('write_path', write_path)
    # print('vocabulary_inv', vocabulary_inv)
    # print('embedding_mat', embedding_mat)
    expanded_seed = []
    vocab_sz = len(vocabulary_inv)
    for j, word_class in enumerate(word_sup_array):
        prob_sup_class = prob_sup_array[j]
        expanded_class = []
        seed_vec = np.zeros(vocab_sz)
        if len(word_class) < sz:
            for i, word in enumerate(word_class):
                seed_vec[word] = prob_sup_class[i]
            expanded = np.dot(embedding_mat.transpose(), seed_vec)
            expanded = np.dot(embedding_mat, expanded)
            word_expanded = sorted(range(len(expanded)), key=lambda k: expanded[k], reverse=True)
            for i in range(sz):
                expanded_class.append(word_expanded[i])
            expanded_seed.append(np.array(expanded_class))
        else:
            expanded_seed.append(word_class)
        if write_path is not None:
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            f = open(write_path + 'class' + str(j) + '_' + str(sz) + '.txt', 'w')
            for i, word in enumerate(expanded_class):
                f.write(vocabulary_inv[word] + ' ')
            f.close()
    # print(expanded_seed)
    return expanded_seed


def label_expansion(class_labels, write_path, vocabulary_inv, embedding_mat):
    print("Retrieving top-t nearest words...")
    n_classes = len(class_labels)
    prob_sup_array = []
    current_szes = []
    all_class_labels = []
    for class_label in class_labels:
        current_sz = len(class_label)
        current_szes.append(current_sz)
        prob_sup_array.append([1/current_sz] * current_sz)
        all_class_labels += list(class_label)
    current_sz = np.min(current_szes)
    while len(all_class_labels) == len(set(all_class_labels)):
        current_sz += 1
        expanded_array = seed_expansion(class_labels, prob_sup_array, current_sz, None, vocabulary_inv, embedding_mat)
        all_class_labels = [w for w_class in expanded_array for w in w_class]


    expanded_array = seed_expansion(class_labels, prob_sup_array, current_sz-1, None, vocabulary_inv, embedding_mat)
    print("expanded array",expanded_array)
    print("Final expansion size t = {}".format(len(expanded_array[0])))

    centers = []
    kappas = []
    print("Top-t nearest words for each class:")
    for i in range(n_classes):
        expanded_class = expanded_array[i]
        vocab_expanded = [vocabulary_inv[w] for w in expanded_class]
        print("Class {}:".format(i))
        print(vocab_expanded)
        expanded_mat = embedding_mat[np.asarray(expanded_class)]
        vmf_soft = VonMisesFisherMixture(n_clusters=1)
        vmf_soft.fit(expanded_mat)
        center = vmf_soft.cluster_centers_[0]
        kappa = vmf_soft.concentrations_[0]
        centers.append(center)
        kappas.append(kappa)

    for j, expanded_class in enumerate(expanded_array):
        if write_path is not None:
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            f = open(write_path + 'class' + str(j) + '.txt', 'w')
            for i, word in enumerate(expanded_class):
                f.write(vocabulary_inv[word] + ' ')
            f.close()
    print("Finished vMF distribution fitting.")
    return expanded_array, centers, kappas


def pseudodocs(word_sup_array, total_num, background_array, sequence_length, len_avg,
                len_std, num_doc, interp_weight, vocabulary_inv, embedding_mat, model, save_dir=None):

    for i in range(len(embedding_mat)):
        embedding_mat[i] = embedding_mat[i] / np.linalg.norm(embedding_mat[i])

    _, centers, kappas = \
    label_expansion(word_sup_array, save_dir, vocabulary_inv, embedding_mat)

    print("Pseudo documents generation...")
    background_vec = interp_weight * background_array
    # if model == 'cnn':
    docs = np.zeros((num_doc*len(word_sup_array), sequence_length), dtype='int32')
    label = np.zeros((num_doc*len(word_sup_array), len(word_sup_array)))
    for i in range(len(word_sup_array)):
        docs_len = len_avg*np.ones(num_doc)
        center = centers[i]
        kappa = kappas[i]
        discourses = sample_vMF(center, kappa, num_doc)
        for j in range(num_doc):
            discourse = discourses[j]
            prob_vec = np.dot(embedding_mat, discourse)
            prob_vec = np.exp(prob_vec)
            sorted_idx = np.argsort(prob_vec)[::-1]
            delete_idx = sorted_idx[total_num:]
            prob_vec[delete_idx] = 0
            prob_vec /= np.sum(prob_vec)
            prob_vec *= 1 - interp_weight
            prob_vec += background_vec
            doc_len = int(docs_len[j])
            docs[i*num_doc+j][:doc_len] = np.random.choice(len(prob_vec), size=doc_len, p=prob_vec)
            label[i*num_doc+j] = interp_weight/len(word_sup_array)*np.ones(len(word_sup_array))
            label[i*num_doc+j][i] += 1 - interp_weight
    # elif model == 'rnn':
    #     docs = np.zeros((num_doc*len(word_sup_array), sequence_length[0], sequence_length[1]), dtype='int32')
    #     label = np.zeros((num_doc*len(word_sup_array), len(word_sup_array)))
    #     doc_len = int(len_avg[0])
    #     sent_len = int(len_avg[1])
    #     for period_idx in vocabulary_inv:
    #         if vocabulary_inv[period_idx] == '.':
    #             break
    #     for i in range(len(word_sup_array)):
    #         center = centers[i]
    #         kappa = kappas[i]
    #         discourses = sample_vMF(center, kappa, num_doc)
    #         for j in range(num_doc):
    #             discourse = discourses[j]
    #             prob_vec = np.dot(embedding_mat, discourse)
    #             prob_vec = np.exp(prob_vec)
    #             sorted_idx = np.argsort(prob_vec)[::-1]
    #             delete_idx = sorted_idx[total_num:]
    #             prob_vec[delete_idx] = 0
    #             prob_vec /= np.sum(prob_vec)
    #             prob_vec *= 1 - interp_weight
    #             prob_vec += background_vec
    #             for k in range(doc_len):
    #                 docs[i*num_doc+j][k][:sent_len] = np.random.choice(len(prob_vec), size=sent_len, p=prob_vec)
    #                 docs[i*num_doc+j][k][sent_len] = period_idx
    #             label[i*num_doc+j] = interp_weight/len(word_sup_array)*np.ones(len(word_sup_array))
    #             label[i*num_doc+j][i] += 1 - interp_weight

    print("Finished Pseudo documents generation.")
    print(docs, label)
    # exit(0)



    return docs, label


def augment(x, sup_idx, total_len):
    print("Labeled documents augmentation...")
    docs = x[sup_idx.flatten()]
    curr_len = len(docs)
    copy_times = int(total_len/curr_len) - 1
    y = np.zeros(len(sup_idx.flatten()), dtype='int32')
    label_nums = [len(seed_idx) for seed_idx in sup_idx]
    cnt = 0
    for i in range(len(sup_idx)):
        y[cnt:cnt+label_nums[i]] = i
        cnt += label_nums[i]

    new_docs = docs
    new_y = y
    for i in range(copy_times):
        new_docs = np.concatenate((new_docs, docs), axis=0)
        new_y = np.concatenate((new_y, y), axis=0)

    pretrain_labels = np.zeros((len(new_y),len(np.unique(y))))
    for i in range(len(new_y)):
        pretrain_labels[i][new_y[i]] = 1.0

    print("Finished labeled documents augmentation.")
    return new_docs, pretrain_labels

def tfidf_pseudolabels(x, word_sup_array, num_doc, vocabulary_inv):
    (m,n) = x.shape
    data = []
    for i in range(m):
        doc = []
        for j in range(n):
            if x[i][j] == 0:
                break
            doc.append(vocabulary_inv[x[i][j]])
        data.append(" ".join(doc))
    # data = np.array(data)

    print("Pseudo labels generation...")
    print(word_sup_array)
    keywords = {}
    for i, word_list in enumerate(word_sup_array):
        keywords[i] = [word for word in word_list]

    # get count features
    count_vectorizer = CountVectorizer(input='content', encoding='ascii',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    min_df=2)
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
    
    class_ids = {}
    for i in range(4):
        class_ids[i] = []

    for i in range(len(data)):
        cur_label = y_pred[i]
        class_ids[cur_label].append(i)

    doc_ids = []
    label = np.zeros((4*num_doc, 4))

    for i in range(4):
        class_total = len(class_ids[i])
        rand_ids = np.random.choice(class_total, num_doc, replace= False)
        sel_doc_ids = [class_ids[i][id] for id in rand_ids]
        doc_ids.extend(sel_doc_ids)

    docs = np.zeros((len(doc_ids), n), dtype=int)
    for i, doc_id in enumerate(doc_ids):
        docs[i] = x[doc_id]
        pred_label = y_pred[doc_id]
        label[i][pred_label] = 1

    # ###### Label all docs ####
    # label = np.zeros((m, 4))
    # for i in range(len(y_pred)):
    #     cur_pred = y_pred[i]
    #     label[i][cur_pred] = 1
    # return x, label
    # ###########################

    return docs, label

def counting_pseudolabels(x, word_sup_array, num_doc, vocabulary_inv):
    (m,n) = x.shape
    data = []
    for i in range(m):
        doc = []
        for j in range(n):
            if x[i][j] == 0:
                break
            doc.append(vocabulary_inv[x[i][j]])
        data.append(" ".join(doc))
    # data = np.array(data)

    print("Pseudo labels generation...")
    print(word_sup_array)
    keywords = {}
    for i, word_list in enumerate(word_sup_array):
        keywords[i] = [word for word in word_list]

    # get count features
    count_vectorizer = CountVectorizer(input='content', encoding='ascii',
                                    decode_error='ignore',
                                    strip_accents='ascii',
                                    min_df=2)
    count_weights = count_vectorizer.fit_transform(data)
    vocabulary = count_vectorizer.vocabulary_

    # get weights for keywords
    keyword_indices = {}
    for class_label, words in keywords.items():
        keyword_indices[class_label] = [vocabulary[w] for w in words]

    class_weights = []
    for class_label in range(4):
        indices = keyword_indices[class_label]
        counts =  np.array(count_weights[:, indices].sum(axis=1)).flatten()
        class_weights.append(counts)
    
    # assign label based on aggregate
    class_weights = np.vstack(class_weights).T

    # exclude words with zero counts
    sum_counts = class_weights.sum(axis = 1)
    excl_ids = sum_counts == 0
    non_zero_ids = [i for i in range(m) if i not in excl_ids]

    y_pred = []
    max_class = np.argmax(class_weights, axis=1)
    for i in range(len(data)):
        y_pred.append(max_class[i])
    
    class_ids = {}
    for i in range(4):
        class_ids[i] = []

    for i in non_zero_ids:
        cur_label = y_pred[i]
        class_ids[cur_label].append(i)

    doc_ids = []
    label = np.zeros((4*num_doc, 4))

    for i in range(4):
        class_total = len(class_ids[i])
        rand_ids = np.random.choice(class_total, num_doc, replace= False)
        sel_doc_ids = [class_ids[i][id] for id in rand_ids]
        doc_ids.extend(sel_doc_ids)

    docs = np.zeros((len(doc_ids), n), dtype=int)
    for i, doc_id in enumerate(doc_ids):
        docs[i] = x[doc_id]
        pred_label = y_pred[doc_id]
        label[i][pred_label] = 1

    # ###### Label all docs ####
    # label = np.zeros((m, 4))
    # for i in range(len(y_pred)):
    #     cur_pred = y_pred[i]
    #     label[i][cur_pred] = 1
    # return x, label
    # ###########################

    return docs, label