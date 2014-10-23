#!/usr/bin/python

"""
Script that parses the CMU Pronunciation Dictionary, and generates the dataset in a nice
numpy format (i.e. in numpy.npz files).
Call :
    parse.py --help
"""
from collections import Counter
import ConfigParser
import cPickle as pickle
import optparse
import os
import re
import time
import sys
import numpy


RND_SEED = 123
TRAIN_SIZE = 0.6
VALID_SIZE = 0.2


def read_words(filename):
    with open(filename, 'rt') as finp:
        for line in finp:
            info = line.split()
            word = info[0].strip()
            word = re.sub('\(.\)', '', word)
            phonemes = info[1:]
            for char in list(word):
                yield char


def read_phones(filename):
    with open(filename, 'rt') as finp:
        for line in finp:
            info = line.split()
            word = info[0].strip()
            word = re.sub('\(.\)', '', word)
            phonemes = info[1:]
            for phone in phonemes:
                yield phone.strip()


def read_all(filename):
    with open(filename, 'rt') as finp:
        for line in finp:
            info = line.split()
            word = info[0].strip()
            word = re.sub('\(.\)', '', word)
            phonemes = map(lambda x: x.strip(), info[1:])
            yield list(word), phonemes


def construct_alphabet(dataset, oov_rate, level):
    filename = dataset
    words = read_words(filename)
    # Order the words
    print ' .. sorting words'
    all_items = Counter(words).items()
    #no_end = [x for x in all_items if x[0] !='\n']
    freqs = sorted([x for x in all_items],
                   key=lambda t: t[1],
                   reverse=True)
    #print ' .. shrinking the vocabulary size'
    # Decide length
    all_freq = float(sum([x[1] for x in freqs]))
    up_to = len(freqs)
    oov = 0.
    remove_word = True
    while remove_word:
        up_to -= 1
        oov += float(freqs[up_to][1])
        if oov / all_freq > oov_rate:
            remove_word = False
    up_to += 1
    freqs = freqs[:up_to]
    words = [x[0] for x in freqs]
    print dict(zip(words, range(up_to)))
    return dict(zip(words, range(up_to))), [x[1]/all_freq for x in freqs], freqs


def construct_phone_dict(dataset, oov_rate, level):
    filename = dataset
    txt = read_phones(filename)
    # Order the words
    print ' .. sorting words'
    all_items = Counter(txt).items()
    freqs = sorted([x for x in all_items],
                   key=lambda t: t[1],
                   reverse=True)
    print ' .. shrinking the vocabulary size'
    # Decide length
    all_freq = float(sum([x[1] for x in freqs]))
    up_to = len(freqs)
    oov = 0.
    remove_word = True
    while remove_word:
        up_to -= 1
        oov += float(freqs[up_to][1])
        if oov / all_freq > oov_rate:
            remove_word = False
    up_to += 1
    freqs = freqs[:up_to]
    words = [x[0] for x in freqs]
    return dict(zip(words, range(up_to))), [x[1]/all_freq for x in freqs], freqs


def grab_text(filename, phone_dict, alph, dtype):
    pairs = [x for x in read_all(filename)]
    numpy.random.seed(RND_SEED)
    pairs = numpy.random.permutation(pairs)

    def construct_lettres(alph, word):
        eol_ind = alph['<eol>']
        arr = numpy.append(numpy.array(map(alph.get, word), dtype=dtype), eol_ind)
        return arr

    def construct_phones(phone_dict, phones):
        eol_ind = phone_dict['<eol>']
        arr = numpy.append(numpy.array(map(phone_dict.get, phones), dtype=dtype), eol_ind)
        return arr

    words = [construct_lettres(alph, word).T for word, _ in pairs]
    phones = [construct_phones(phone_dict, phones).T for _, phones in pairs]
    return words, phones, len(pairs)


def grab_text_binary(filename, phone_dict, alph, dtype):
    pairs = [x for x in read_all(filename)]
    numpy.random.seed(RND_SEED)
    numpy.random.permutation(pairs)

    def construct_lettres(alph, word):
        alph_size = len(alph)
        word_size = len(word)
        arr = numpy.zeros((alph_size, word_size), dtype=dtype)
        ind = map(alph.get, word)
        for i in xrange(word_size):
            arr[ind[i], i] = 1.0
        return arr

    def construct_phones(phone_dict, phones):
        vocab_size = len(phone_dict)
        seq_size = len(phones)
        arr = numpy.zeros((vocab_size, seq_size), dtype=dtype)
        ind = map(phone_dict.get, phones)
        for i in xrange(seq_size):
            arr[ind[i], i] = 1.0
        return arr

    words = [construct_lettres(alph, word).T for word, _ in pairs]
    phones = [construct_phones(phone_dict, phones).T for _, phones in pairs]
    return words, phones, len(pairs)


def main(parser):
    o, _ = parser.parse_args()
    dataset = '/data/lisatmp3/serdyuk/cmudict/cmudict.0.7a.agg'
    print ' .. constructing the vocabulary'
    alph, freqs, freq_wd = construct_alphabet(dataset, o.oov_rate, o.level)
    alph['<eol>'] = len(alph)

    phone_vocab, phone_freqs, phone_freq_wd = construct_phone_dict(dataset, o.oov_rate, o.level)
    phone_vocab['<eol>'] = len(phone_vocab)

    if o.oov == '-1':
        oov_default = -1
    else:
        oov_default = len(phone_vocab)
    print ' .. constructing train set'
    data_words, data_phones, size = grab_text(dataset, phone_vocab, alph, o.dtype)
    train_words = data_words[:int(TRAIN_SIZE * size)]
    train_phones = data_phones[:int(TRAIN_SIZE * size)]

    print ' .. constructing valid set'
    valid_words = data_words[int(TRAIN_SIZE * size):int((TRAIN_SIZE + VALID_SIZE) * size)]
    valid_phones = data_phones[int(TRAIN_SIZE * size):int((TRAIN_SIZE + VALID_SIZE) * size)]
    
    print ' .. constructing test set'
    test_words = data_words[int((TRAIN_SIZE + VALID_SIZE) * size):]
    test_phones = data_phones[int((TRAIN_SIZE + VALID_SIZE) * size):]

    print ' .. saving data'
    
    data = pickle.dumps(dict(
        train_words=train_words,
        valid_words=valid_words,
        test_words=test_words,
        train_phones=train_phones,
        valid_phones=valid_phones,
        test_phones=test_phones,
        phone_dict_size=len(phone_vocab),
        phone_dict=phone_vocab,
        alphabet_size=len(alph),
        alphabet=alph
        ))
    with open(o.dest, "wt") as fout:
        fout.write(data)
    print '... Done'


def get_parser():
    usage = """
This script parses the CMU Pronunciation Dictionary from
http://www.speech.cs.cmu.edu/cgi-bin/cmudict, and generates more numpy friendly
format of the dataset. Please use this friendly formats as temporary forms
of the dataset (i.e. delete them after you're done).

The script will save the entire file into a numpy .npz file. The file will
contain the following fields:
    'train_words' : a list of arrays where each element (word) is
              represented by an array of indexes from 0 to alphabet size. 
              It is the training data.
    'train_phones' : a list of arrays where each element (sequence of phones) is
              represented by an array of indexes from 0 to phone vocabulary size. 
              It is the training data.
    'test_words' : a list of arrays where each element (word) is represented by an
             array of indexes from 0 to alphabet size. This is the test value.
    'test_phones' : a list of arrays where each element (sequence of phones) is 
             represented by an array of indexes from 0 to phone vocabulary size. 
             This is the test value.
    'valid_words' : a list of arrays where each element (word) is represented by an
             array of indexes from 0 to alphabet size. This is the validation set.
    'valid_phones' : a list of arrays where each element (sequence of phones) is 
             represented by an array of indexes from 0 to phone vocabulary size. 
             This is the validation set.
    'phone_dict_size' : The size of the phone dictionary.
    'phone_dict' : The phone dictionary.
    'alphabet_size' : The size of the alphabet.
    'alphabet' : The alphabet.
    """
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--dest',
                      dest='dest',
                      help=('Where to save the processed dataset (i.e. '
                            'under what name and at what path)'),
                      default='tmp_data.pkl')
    parser.add_option('--level',
                      dest='level',
                      help=('Processing level. Either `words` or `letter`. '
                            'If set to word, the result dataset has one '
                            'token per word, otherwise a token per letter'),
                      default='words')
    parser.add_option('--n_chains',
                      dest='n_chains',
                      type="int",
                      help=('Number of parallel chains for the training '
                            'data. The way it works, is that it takes the '
                            'training set and divides it in `n_chains` that '
                            'should be processed in parallel by your model'),
                      default=1)
    # Dataset is already split !
    """
    parser.add_option('--train_size',
                      dest='train_size',
                      type="string",
                      help=('number of samples in the training set. please '
                            'use something like 10m (for 10 millions), 10k '
                            'for (10 thousands). if no letter at the end, '
                            'it is assumed to be an exact number of samples')
                      default='90M')
    parser.add_option('--valid_size',
                      dest='valid_size',
                      type='string',
                      help=('number of samples in the validation set. please '
                            'use something like 10m (for 10 millions), 10k '
                            'for (10 thousands). if no letter at the end, '
                            'it is assumed to be an exact number of samples')
                      default='5M')
    """
    parser.add_option('--oov_index',
                      dest='oov',
                      type='string',
                      help=('index for oov words (in case of word level). '
                            'The accepted values can be `-1`, `last`'),
                      default='-1')
    parser.add_option('--oov_rate',
                      dest='oov_rate',
                      type='float',
                      help=('Defines dictionary size. If for example '
                            'oov_rate is set to 0.01 (meaning 10%) it means '
                            'that we can shrink our dictionary such that '
                            'remaining unrepresented words of the **train** '
                            'set is less then 10%. If set to 0, all words in '
                            'the training set will be added to the '
                            'dictionary'),
                      default=0.)
    parser.add_option('--dtype',
                      dest='dtype',
                      help='dtype in which to store data',
                      default='int32')
    return parser

if __name__ == '__main__':
    main(get_parser())
