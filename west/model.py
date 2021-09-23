import numpy as np

np.random.seed(1234)
import os
from time import time
import csv
import keras.backend as K
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(
# intra_op_parallelism_threads=30, inter_op_parallelism_threads=30)))
from keras.engine.topology import Layer
from keras.layers import Dense, Input, Convolution1D, Embedding, \
    GlobalMaxPooling1D, GRU, TimeDistributed, LSTM, Dropout, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from sklearn.metrics import classification_report
from keras import regularizers, constraints
from keras.initializers import RandomUniform
from sklearn.metrics import f1_score

from keras.callbacks import EarlyStopping


def f1(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    return f1_macro, f1_micro


def ConvolutionLayer(input_shape,
                     n_classes,
                     filter_sizes=[2, 3, 4, 5],
                     num_filters=16,
                     word_trainable=False,
                     vocab_sz=None,
                     embedding_matrix=None,
                     word_embedding_dim=100,
                     hidden_dim=16,
                     act='relu',
                     init='ones'):
    x = Input(shape=(input_shape,), name='input')
    z = Embedding(vocab_sz, word_embedding_dim, input_length=(input_shape,),
                  name="embedding",
                  weights=[embedding_matrix], trainable=word_trainable)(x)
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation=act,
                             strides=1,
                             kernel_initializer=init)(z)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Dense(hidden_dim, activation="relu")(z)
    y = Dense(n_classes, activation="softmax")(z)
    return Model(inputs=x, outputs=y, name='classifier')


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 init='glorot_uniform', bias=True, **kwargs):

        self.supports_masking = True
        self.init = init

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def HierAttLayer(input_shape, n_classes, word_trainable=False, vocab_sz=None,
                 embedding_matrix=None, word_embedding_dim=100, gru_dim=100,
                 fc_dim=100):
    sentence_input = Input(shape=(input_shape[1],), dtype='int32')
    embedded_sequences = Embedding(vocab_sz,
                                   word_embedding_dim,
                                   input_length=input_shape[1],
                                   weights=[embedding_matrix],
                                   trainable=word_trainable)(sentence_input)
    l_gru = GRU(gru_dim, return_sequences=True)(embedded_sequences)
    l_dense = TimeDistributed(Dense(fc_dim))(l_gru)
    l_att = AttentionWithContext()(l_dense)
    y = Dense(n_classes, activation='softmax')(l_att)
    return Model(inputs=sentence_input, outputs=y, name='classifier')



def LstmLayer(input_shape, n_classes, word_trainable=False, vocab_sz=None,
                 embedding_matrix=None, word_embedding_dim=100, num_lstm=300,
                 rate_drop_lstm = 0.25, fc_dim=100, num_dense=256):
    sentence_input = Input(shape=(input_shape[1],), dtype='int32')
    embedded_sequences = Embedding(vocab_sz,
                                   word_embedding_dim,
                                   input_length=input_shape[1],
                                   weights=[embedding_matrix],
                                   trainable=word_trainable)(sentence_input)
    l_lstm = Bidirectional(LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm,return_sequences=True))(embedded_sequences)
    # l_dense = Dense(num_dense, activation='relu')(l_lstm)
    l_dense = TimeDistributed(Dense(fc_dim))(l_lstm)
    l_att = AttentionWithContext()(l_dense)
    l_dense = Dense(fc_dim, activation='softmax')(l_att)
    l_dropout = Dropout(rate_drop_lstm)(l_dense)
    l_bn = BatchNormalization()(l_dropout)
    y = Dense(n_classes, activation='softmax')(l_bn)

    # l_drop = Dropout(rate_drop_dense)(merged)
    return Model(inputs=sentence_input, outputs=y, name='classifier')


    




class WSTC(object):
    def __init__(self,
                 input_shape,
                 n_classes=None,
                 init=RandomUniform(minval=-0.01, maxval=0.01),
                 y=None,
                 model='cnn',
                 vocab_sz=None,
                 word_embedding_dim=100,
                 embedding_matrix=None
                 ):

        super(WSTC, self).__init__()

        self.input_shape = input_shape
        self.y = y
        self.n_classes = n_classes
        if model == 'cnn':
            self.classifier = ConvolutionLayer(self.input_shape[1],
                                               n_classes=n_classes,
                                               vocab_sz=vocab_sz,
                                               embedding_matrix=embedding_matrix,
                                               word_embedding_dim=word_embedding_dim,
                                               init=init)
        elif model == 'rnn':
            self.classifier = HierAttLayer(self.input_shape,
                                           n_classes=n_classes,
                                           vocab_sz=vocab_sz,
                                           embedding_matrix=embedding_matrix,
                                           word_embedding_dim=word_embedding_dim)
        elif model == 'lstm':
            self.classifier = LstmLayer(self.input_shape,
                                           n_classes=n_classes,
                                           vocab_sz=vocab_sz,
                                           embedding_matrix=embedding_matrix,
                                           word_embedding_dim=word_embedding_dim)

        self.model = self.classifier
        self.classifier_name = model
        self.sup_list = {}

    def pretrain(self, x, pretrain_labels, sup_idx=None, optimizer='adam',
                 loss='kld', epochs=200, batch_size=256, save_dir=None):

        self.classifier.compile(optimizer=optimizer, loss=loss)
        print("\nNeural model summary: ")
        self.model.summary()

        if sup_idx is not None:
            for i, seed_idx in enumerate(sup_idx):
                for idx in seed_idx:
                    self.sup_list[idx] = i

        # begin pretraining
        t0 = time()
        print('\nPretraining...')
        if self.classifier_name == 'cnn':
            history = self.classifier.fit(x, pretrain_labels,
                                          batch_size=batch_size,
                                          validation_split=0.2,
                                          epochs=epochs,
                                          callbacks=[EarlyStopping(
                                              monitor='val_loss',
                                              restore_best_weights=True)])
        else:
            history = self.classifier.fit(x, pretrain_labels,
                                          batch_size=batch_size,
                                          validation_split=0.2,
                                          epochs=epochs,
                                          callbacks=[
                                              EarlyStopping(monitor='val_loss',
                                                            restore_best_weights=True,
                                                            patience=5)])
        print(history.history.keys())
        print('Pretraining time: {:.2f}s'.format(time() - t0))

        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            model_json = self.classifier.to_json()
            with open(save_dir + "/model.json", "w") as json_file:
                json_file.write(model_json)
            self.classifier.save_weights(save_dir + "/model.h5")
            print('Pretrained model saved to {}/model.h5'.format(save_dir))
        self.pretrained = True

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    def target_distribution(self, q, power=2):
        weight = q ** power / q.sum(axis=0)
        p = (weight.T / weight.sum(axis=1)).T
        for i in self.sup_list:
            p[i] = 0
            p[i][self.sup_list[i]] = 1
        return p

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=10, batch_size=256, tol=0.1, power=2,
            update_interval=140, save_dir=None, save_suffix=''):

        print('Update interval: {}'.format(update_interval))

        pred = self.classifier.predict(x)
        y_pred = np.argmax(pred, axis=1)
        print(classification_report(y, y_pred))
        y_pred_last = np.copy(y_pred)

        # logging file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(
            save_dir + '/self_training_log_{}.csv'.format(save_suffix), 'w')
        logwriter = csv.DictWriter(logfile,
                                   fieldnames=['iter',
                                               'acc',
                                               'f1_macro',
                                               'f1_micro',
                                               'f1_normal',
                                               'f1_spam',
                                               'f1_abusive',
                                               'f1_hateful'])
        logwriter.writeheader()

        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            print('\nIter {}: '.format(ite), end='')

            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                y_pred = q.argmax(axis=1)
                p = self.target_distribution(q, power)

                if y is not None:
                    f1_macro, f1_micro = np.round(f1(y, y_pred), 5)
                    report = classification_report(y, y_pred, output_dict=True)
                    print(classification_report(y, y_pred))
                    logdict = dict(iter=ite,
                                   acc=round(report['accuracy'], 5),
                                   f1_macro=round(
                                       report['macro avg']['f1-score'], 5),
                                   f1_micro=round(report['weighted avg'][
                                                      'f1-score'], 5),
                                   f1_normal=round(report['0']['f1-score'], 5),
                                   f1_spam=round(report['1']['f1-score'], 5),
                                   f1_abusive=round(report['2']['f1-score'],
                                                    5),
                                   f1_hateful=round(report['3']['f1-score'],
                                                    5))
                    logwriter.writerow(logdict)
                    print('f1_macro = {}, f1_micro = {}'.format(f1_macro,
                                                                f1_micro))

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float) \
                              / \
                              y_pred.shape[0]
                print('Number of documents with label changes: {}'.format(
                    np.sum(y_pred != y_pred_last)))
                print('Fraction of documents with label changes: {} %'.format(
                    np.round(delta_label * 100, 3)))
                if ite > 0 and delta_label < tol / 100:
                    print('\nFraction: {} % < tol: {} %'.format(
                        np.round(delta_label * 100, 3), tol))
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break
                y_pred_last = np.copy(y_pred)

            # train on batch
            idx = index_array[index * batch_size: min((index + 1) * batch_size,
                                                      x.shape[0])]

            batch_loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            print("Training loss", str(batch_loss))
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            ite += 1

        logfile.close()

        if save_dir is not None:
            self.model.save_weights(save_dir + '/final.h5')
            print("Final model saved to: {}/final.h5".format(save_dir))
        return self.predict(x)
