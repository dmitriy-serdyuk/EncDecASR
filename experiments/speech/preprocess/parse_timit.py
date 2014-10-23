#!/usr/bin/python

__author__ = 'serdyuk'

import cPickle as pickle
import sys, os
import optparse
import numpy as np


def construct_alphabet(dataset):
    with open(dataset + 'readable/words.pkl', 'rb') as fin:
        words = pickle.load(fin)
    alph = {' ': 0}
    k = 1
    for word in words:
        for char in list(word):
            if not char in alph:
                alph[char] = k
                k += 1
    alph['<eol>'] = k
    return alph


def grab_text(dataset, phonem_dict, alph):
    words = []
    phones = []
    for name in os.walk(dataset):
        if name[1] != []:
            continue
        dir = name[0]
        filenames = set([os.path.splitext(filename)[0] for filename in os.listdir(dir)])
        for filename in filenames:
            sent = []
            with open(os.path.join(dir, filename + '.WRD')) as fin:
                for line in fin:
                    word = line.split()[2].strip()
                    sent += [word]
            words += [np.array(map(alph.get, ' '.join(sent)) + [alph['<eol>']], dtype='int64')]
            with open(os.path.join(dir, filename + '.PHN')) as fin:
                ph_arr = []
                for line in fin:
                    phone = line.split()[2].strip()
                    ph_arr += [phone]
            phones += [np.array(map(phonem_dict.get, ph_arr) + [phonem_dict['<eol>']], dtype='int64')]

    return words, phones


def main(parser):
    o, _ = parser.parse_args()
    dataset = '/data/lisa/data/timit/'
    print ' .. constructing the alphabet'
    alph = construct_alphabet(dataset)

    print ' .. extracting phones'
    with open(dataset + 'readable/phonemes.pkl', 'rb') as fin:
        phonemes = pickle.load(fin)
    phone_dict = {phone: ind for ind, phone in enumerate(phonemes)}
    phone_dict['<eol>'] = len(phone_dict)

    print ' .. constructing train set'
    train_words, train_phones = grab_text(dataset + 'raw/TIMIT/TRAIN/', phone_dict, alph)

    print ' .. constructing test set'
    test_words, test_phones = grab_text(dataset + 'raw/TIMIT/TEST/', phone_dict, alph)

    print ' .. saving data'

    data = pickle.dumps(dict(
        train_words=train_words,
        test_words=test_words,
        train_phones=train_phones,
        test_phones=test_phones,
        phone_dict_size=len(phone_dict),
        phone_dict=phone_dict,
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
