#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint

import numpy

from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.trainer.SGD import SGD as SGD
from groundhog.trainer.SGD_momentum import SGD as SGD_momentum
from groundhog.mainLoop import MainLoop
from encdec import RNNEncoderDecoder #, prototype_state, get_batch_iterator
import experiments.speech
from iterators import CMUIterator, get_cmu_batch_iterator


logger = logging.getLogger(__name__)


class RandomSamplePrinter(object):

    def __init__(self, state, model, train_iter):
        self.state = state
        self.model = model
        self.train_iter = train_iter
        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

    def __call__(self):
        def cut_eol(words):
            for i, word in enumerate(words):
                if words[i] == '<eol>':
                    return words[:i + 1]
            raise Exception("No end-of-line found")

        sample_idx = 0
        while sample_idx < self.state['n_examples']:
            batch = self.train_iter.next(peek=True)
            
            xs, ys = batch['x'], batch['y']
            for seq_idx in range(xs.shape[1]):
                if sample_idx == self.state['n_examples']:
                    break

                x, y = xs[:, seq_idx], ys[:, seq_idx]
                x_words = cut_eol(map(lambda w_idx: self.model.word_indxs_src[w_idx], x))
                y_words = cut_eol(map(lambda w_idx: self.model.word_indxs[w_idx], y))
                if len(x_words) == 0:
                    continue

                print "Input: {}".format(" ".join(x_words))
                print "Target: {}".format("".join(y_words))
                self.model.get_samples(self.state['seqlen'] + 1, self.state['n_samples'], x[:len(x_words)])
                sample_idx += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_speech_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    return parser.parse_args()


def main():
    args = parse_args()

    state = getattr(experiments.speech, args.proto)()
    if args.state:
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    logging.basicConfig(level=getattr(logging, state['level']),
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug("State:\n{}".format(pprint.pformat(state)))

    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, args.skip_init)
    enc_dec.build()
    pronunciation_model = enc_dec.create_pronunciation_model()

    logger.debug("Load data")
    train_data = get_cmu_batch_iterator(state=state, rng=rng, logger=logger, subset='train')
    logger.debug("Load validation data")
    valid_data = get_cmu_batch_iterator(state=state, rng=rng, logger=logger, subset='valid')
    logger.debug("Compile trainer")
    if state['algo'] == 'SGD_adadelta':
        algo = SGD_adadelta(pronunciation_model, state, train_data)
    elif state['algo'] == 'SGD':
        algo = SGD(pronunciation_model, state, train_data)
    elif state['algo'] == 'SGD_momentum':
        algo = SGD_momentum(pronunciation_model, state, train_data)
    else:
        raise Exception("Illegal training algorithm")

    logger.debug("Run training")
    main = MainLoop(train_data, valid_data, test_data=None, model=pronunciation_model, algo=algo,
                    state=state, channel=None,
                    reset=state['reset'],
                    hooks=[RandomSamplePrinter(state, pronunciation_model, train_data)]
                    if state['hookFreq'] >= 0
                    else None
    )
    if state['reload']:
        main.load()
    if state['loopIters'] > 0:
        main.main()

if __name__ == "__main__":
    main()
