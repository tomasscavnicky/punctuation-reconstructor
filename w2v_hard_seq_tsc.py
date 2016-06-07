# original script autor: Karel Benes
# modified by: Tomas Scavnicky

import IPython

import sys
import random
import argparse
import math

import numpy as np
import theano
import theano.tensor as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# tieto su tiez custom
from seqgen import SeqGen
from word2vec import Word2Vec

from keras.models import Sequential
from keras.layers.core import Dense, TimeDistributedDense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.utils.data_utils import get_file
from keras.optimizers import Adagrad, SGD
from keras.callbacks import Callback
from keras.utils.generic_utils import Progbar

NO_PUNCT = ''
PERIOD='</s>'
COMMA='</comma>'

ps = {NO_PUNCT:0, PERIOD:1, COMMA:2}



def indices_to_one_hot(inds, width):
   one_hot = np.zeros((len(inds), width), dtype=np.bool) 
   one_hot[np.arange(len(inds)), inds] = 1
   return one_hot


def hard_confusion_matrix(predictions, targets, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int)
    for y, t in zip(predictions, targets):
        cm[t][y] += 1

    return cm

def seq_to_w2v_fea(inds):
    vector_repr = [word2vec.w2v(w) for w in inds]
    np_repr = np.asarray(vector_repr, dtype=np.float32)
    return np_repr

def prep_xy_from_data(batch_data, input_width, labels_width):
    # get the sequences
    batch_inds = map(lambda sample: sample[0], batch_data)
    X = np.asarray([seq_to_w2v_fea(inds) for inds in batch_inds])

    # get the labels of sequences
    batch_labels = map(lambda sample: sample[1], batch_data)
    y = np.asarray(indices_to_one_hot(batch_labels, labels_width))

    return X, y, batch_labels

def show_scores(pos_scores, neg_scores, title="Figure"):
    if args.img_dir == None:
        return

    point_scores_pos = [x[1] for x in pos_scores]
    point_scores_neg = [x[1] for x in neg_scores]

    plt.figure()
    x,bins,p = plt.hist([point_scores_pos, point_scores_neg], bins=20,
            stacked=False, label=("Puncted samples", "Nonpuncted samples"),
            range=(0.0,1.0), histtype='step')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.savefig(args.img_dir + '/' + title + ".pdf",bbox_inches='tight')


def get_f1_score(conf_matrix):
    corr_punct = float(conf_matrix[1][1])
    misses = float(conf_matrix[1][0])
    false_positives = float(conf_matrix[0][1])

    try:
        r = corr_punct/(corr_punct + misses)
        p = corr_punct/(corr_punct + false_positives) 
        f1 = 2/(1/r + 1/p)
    except ZeroDivisionError:
        f1 = float("nan")

    return f1




def run_epoch(model, train_pcts, train_npcts, valid_data, batch_size, epoch_name, w2v):
    epoch_results = []

    print "Training: "
    train_scores_punct = []
    train_scores_nonpunct = []
    train_cm = np.zeros((len(ps), len(ps)), dtype=np.int)
    n_batches = 2*len(train_pcts)//batch_size
    print "n_batches:", n_batches
    if args.verbose == 1:
        train_progbar = Progbar(target=n_batches)
    n_neg_samples = args.neg_samples_ratio*(batch_size/2)
    for batch_no in range(n_batches):
        # create batch from examples of punctuated and non-punctuated data
        batch_data = []
        batch_data.extend(train_pcts[batch_no*(batch_size/2) : (batch_no+1)*(batch_size/2)])
        batch_data.extend(train_npcts[batch_no*n_neg_samples : (batch_no+1)*n_neg_samples]) # twice as many non-punct examples as punct ones

        X, y, batch_labels = prep_xy_from_data(batch_data, len(w2i), len(ps))

        preds = model.predict_classes(X, batch_size=batch_size, verbose=0)
        train_cm += hard_confusion_matrix(preds, batch_labels, len(ps))
        scores = model.predict(X, batch_size=batch_size, verbose=0)
        pos_res, neg_res = sep_pos_neg_res(scores.tolist(), batch_labels)
        train_scores_punct += pos_res
        train_scores_nonpunct += neg_res

        (loss, acc) = model.train_on_batch(X, y, accuracy=True)
        if args.verbose == 1:
            train_progbar.add(1, [('loss', loss), ('acc', acc)])
        epoch_results.append((loss, acc))

    print "\taverage accuracy:", np.mean(map(lambda r: r[1], epoch_results))
    print "\taverage loss:", np.mean(map(lambda r: r[0], epoch_results))
    print "\tconfusion matrix:"
    print "\tF1-score:", get_f1_score(train_cm)
    print train_cm
    show_scores(train_scores_punct, train_scores_nonpunct, epoch_name + " training")
    print "\tw2v experience:", w2v.experience()
    w2v.reset_experience()

    print "Validation:"
    random.shuffle(valid_data)
    valid_results = []
    valid_scores_punct = []
    valid_scores_nonpunct = []
    valid_cm = np.zeros((len(ps), len(ps)), dtype=np.int)
    n_batches = len(valid_data)//batch_size
    if args.verbose == 1:
        valid_progbar = Progbar(target=n_batches)
    for batch_no in range(n_batches):
        batch_data = valid_data[batch_no*batch_size:(batch_no+1)*batch_size]

        X, y, batch_labels = prep_xy_from_data(batch_data, len(w2i), len(ps))

        preds = model.predict_classes(X, batch_size=batch_size, verbose=0)
        valid_cm += hard_confusion_matrix(preds, batch_labels, len(ps))

        scores = model.predict(X, batch_size=batch_size, verbose=0)
        pos_res, neg_res = sep_pos_neg_res(scores.tolist(), batch_labels)
        valid_scores_punct += pos_res
        valid_scores_nonpunct += neg_res

        (loss, acc) = model.test_on_batch(X, y, accuracy=True)
        if args.verbose == 1:
            valid_progbar.add(1, [('loss', loss), ('acc', acc)])
        valid_results.append((loss, acc))

    print "\taverage accuracy:", np.mean(map(lambda r: r[1], valid_results))
    print "\taverage loss:", np.mean(map(lambda r: r[0], valid_results))
    print "\tconfusion matrix:"
    print "\tF1-score:", get_f1_score(valid_cm)
    print valid_cm
    show_scores(valid_scores_punct, valid_scores_nonpunct, epoch_name + " validation")
    print "\tw2v experience:", w2v.experience()
    w2v.reset_experience()





def get_new_model(in_width, lstm_hidden, out_width, batch_size):
    model = Sequential()
    if args.time_dist_dense:
        model.add(LSTM(in_width, lstm_hidden, return_sequences=True))
        model.add(TimeDistributedDense(lstm_hidden, out_width))
    else:
        if args.rnn:
            model.add(SimpleRNN(in_width, lstm_hidden, return_sequences=False))
        else:
            print "in_width: ", str(in_width)
            print "lstm_hidden: ", str(lstm_hidden[0])
            model.add(LSTM(lstm_hidden[0] , batch_input_shape=(None, batch_size, in_width), return_sequences=False))
        model.add(Dropout(args.dropout))
        model.add(Dense(lstm_hidden[0]))
    model.add(Activation('softmax'))
    optimizer = Adagrad(clipnorm=args.clipnorm)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model






if __name__ == '__main__':
    print "start"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to", 
                            help="which file to store model into")
    parser.add_argument("--w2v_file", nargs=1,
                            help="file word2vec specs")
    parser.add_argument("--vocab_file", 
                            help="file containing all relevant words")
    parser.add_argument("--img_dir", 
                            help="directory to save images in")
    parser.add_argument("--rnn", action='store_true',
                            help="use rnn instead of LSTM")
    parser.add_argument("--shuffle_by_sort", action='store_true',
                            help="sort the data according to their unigram difficulty")
    parser.add_argument("--time_dist_dense", action='store_true',
                            help="use all lstm values as input to softmax")
    parser.add_argument("--n_epochs", type=int, default=0,
                            help="number of training epochs")
    parser.add_argument("--n_hidden", type=int, nargs=1,
                            help="number of hidden units")
    parser.add_argument("--batch_size", type=int, default=32,
                            help="number of examples in batch")
    parser.add_argument("--sample_len", type=int, default=10,
                            help="length of training examples")
    parser.add_argument("--target_ind", type=int, default=-1,
                            help="which value use as the target")
    parser.add_argument("--neg_samples_ratio", type=int, default=1,
                            help="#neg examples/#pos examples per batch")
    parser.add_argument("--epochal_increase", type=float, default=1.0,
                            help="portion of data to add into the training set each epoch")
    parser.add_argument("--clipnorm", type=float, default=1000,
                            help="optimizer option")
    parser.add_argument("--verbose", type=int, default=2,
                            help="directly passed to Keras. O - silent, 1 - ProgBar, 2 - line per epoch")
    parser.add_argument("--dropout", type=float, default=0,
                            help="dropout on top of recursive layer")
    parser.add_argument("--test_on", help="file with test examples")
    parser.add_argument("--blind_test_on", help="file with text to punctate")
    parser.add_argument("train_file", nargs=1, help="file with training data")
    args = parser.parse_args()


    # toto nieviem co presne robi
    if args.target_ind == args.sample_len:
        args.target_ind = True # sequential target


    print "User config:", args

    print "reading w2v file..."
    word2vec = Word2Vec(args.w2v_file[0])

    print "vectors are", word2vec.vector_len(), "elements long"

    print "reading input..."
    text = open(args.train_file[0]).read()

    print "constructing vocabulary..."
    words = set(text.split())


    print "constructing translation lambdas..."
    w2i = dict((w, i) for i, w in enumerate(words))
    i2w = dict((i, w) for i, w in enumerate(words))

    w2i_f = lambda w: w2i[w]
    w2t_f = lambda w: ps[w] if w in ps else 0
    identity = lambda x: x


    if args.n_epochs > 0:
        print "creating examples..."
        npcts = []
        pcts = []
        index = 0
        for seq, target in SeqGen(text.split(), identity, w2t_f, args.sample_len, "<unk>", args.target_ind):
            if index < 50:
                print seq, target
                index += 1
            if target == ps[NO_PUNCT]:
                npcts += [(seq, target)]
            else:
                pcts += [(seq, target)]

    # for i in range(1,30):
    #     print npcts[i]
    for i in range(1,30):
        print pcts[i]


    print "shuffling examples..."
    random.shuffle(pcts)
    random.shuffle(npcts)


    print "splitting to training set to training an validation..."
    valid_ratio = 0.1
    train_pcts = pcts[:-int(valid_ratio*len(pcts))]
    valid_pcts = pcts[-int(valid_ratio*len(pcts)):]
    train_npcts = npcts[:-int(valid_ratio*len(npcts))]
    valid_npcts = npcts[-int(valid_ratio*len(npcts)):]

    valid_data = valid_pcts + valid_npcts



    print 'Total number of distinc words:', len(w2i)
    print 'Training dataset:'
    print '\tNon punctuation examples:', len(train_npcts)
    print '\tPunctuation example:', len(train_pcts)
    
    print 'Validation dataset:'
    print '\tNon punctuation examples:', len(valid_npcts)
    print '\tPunctuation example:', len(valid_pcts)


    print 'Building model...'

    model = get_new_model(word2vec.vector_len(), args.n_hidden, len(ps), args.batch_size)


    if args.n_epochs > 0:
        print 'Fitting to data...'

        if args.shuffle_by_sort:
            train_pcts.sort(key = lambda seq: unigram_difficulty(seq[0], unigrams))
            train_npcts.sort(key = lambda seq: unigram_difficulty(seq[0], unigrams))


        for epoch_no in range(args.n_epochs):
            print "\nEpoch number:", epoch_no

            this_epoch_portion = min([(epoch_no+1)*args.epochal_increase, 1.0])
    

            epoch_train_pos = train_pcts[:int(this_epoch_portion*len(train_pcts))]
            epoch_train_neg = train_npcts[:int(this_epoch_portion*len(train_npcts))]

            random.shuffle(epoch_train_pos)
            random.shuffle(epoch_train_neg)

            run_epoch(model, epoch_train_pos, epoch_train_neg, valid_data, batch_size=args.batch_size, epoch_name="Epoch " + str(epoch_no+1), w2v=word2vec)

        if args.model_to != None:
            model.save_weights(args.model_to, overwrite=True)
        sys.stderr.close()
        sys.stderr = sys.__stderr__






