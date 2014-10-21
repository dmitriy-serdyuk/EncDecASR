'''
Created on Aug 21, 2014

@author: chorows
'''

import logging
logger = logging.getLogger(__file__)

from itertools import izip
from collections import OrderedDict, namedtuple

import numpy as np
import cPickle

from pylearn2.utils.rng import make_np_rng


class AbstractDataIterator(object):
    """
The AbstractDataIterator implements the iterator protocol with the following additions:
1. I returns namedtuples of numpy arrays only
2. The namedtuples are created by self.OutputClass
3. The names of sources it carries can be queried by self.source_names property. 
   They correspond to the fields of the namedtuple
  
Each element returned by the iterator is one batch of data. Individual examples in the batch are 
indexed by the first dimension, the interpretatoin of the other dimensions is left to the processing functions.
   
The idea is that The data preprocessing pipeline will be created by plugging those iterators, 
specific sources can be always queried by name.
    """
    
    @staticmethod
    def create_output_class(source_names):
        return namedtuple(
                    typename='OutTuple_'+'_'.join(source_names), 
                    field_names=source_names)
    
    def __init__(self, sources = None, source_names=None, OutputClass=None, properties=None, **kwargs):
        super(AbstractDataIterator, self).__init__(**kwargs)
        if properties is None:
            properties = {}
        num_args = sum(i is not None for i in [sources, source_names, OutputClass])
        if num_args != 1:
            raise Exception("Must provide exactly one of sources, source_names, OutClass! (provided %d)" % (num_args, ))
        if OutputClass is not None:
            self.OutputClass = OutputClass
        else:
            if source_names is None:
                source_names=sources.keys()
                properties['sources'] = sources
            self.OutputClass = AbstractDataIterator.create_output_class(source_names)
                    
        self.properties=properties

    #Mandatory functions to override:
    def next(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def start(self, start_offset=0):
        raise NotImplementedError()
    
    #Normal operation
    def make(self, iterable):
        return self.OutputClass._make(iterable)
    
    @property
    def source_names(self):
        return self.OutputClass._fields
    
    @property
    def sources(self):
        sources = self.properties['sources']
        ret = OrderedDict()
        for sn in self.source_names:
            ret[sn] = sources[sn]
        return ret
    
    @property
    def stochastic(self):
        return self.properties.get('stochastic', None)
    
    @property
    def num_examples(self):
        return self.properties.get('num_examples', float('nan'))
    
    @property
    def num_batches(self):
        return self.properties.get('num_batches', float('nan'))
    
    @property
    def batch_size(self):
        return self.properties.get('batch_size', float('nan'))
    
    @property
    def uneven(self):
        return self.properties.get('uneven', True)
    
    def __repr__(self):
        return "DataIterator returning %s" % (self.source_names, )
        
    def __iter__(self):
        return self
    

class AbstractWrappedIterator(AbstractDataIterator):
    """
An abstract class for a iterator that wraps other iterators and transforms its data
    """
    def __init__(self, iterator, **kwargs):
        super(AbstractWrappedIterator, self).__init__(OutputClass=iterator.OutputClass, **kwargs)
        self.iterator = iterator.__iter__()
        self.properties = dict(self.iterator.properties)
        
        #unless a class restores it, we are better off assuming they have changed??
        #self.properties.sources=None  


class DataIterator(AbstractDataIterator):
    """
An iterator that wraps a Python Iterator
    """
    def __init__(self, iterator, source_names, **kwargs):
        super(DataIterator, self).__init__(source_names=source_names, **kwargs)
        self.iterator = iterator
    
    def next(self):
        return self.make(self.iterator.next())


class NamedTupleIterator(AbstractDataIterator):
    """
An iterator that wraps a Python iterator that already returns named tuples
    """
    def __init__(self, iterator, OutputClass, **kwargs):
        super(NamedTupleIterator, self).__init__(OutputClass=OutputClass, **kwargs)
        self.iterator = iterator
    
    def next(self):
        return self.iterator.next()


class TransformingIterator(AbstractWrappedIterator):
    """
An iterator that transforms the batches that pass through it. 
    """
    def __init__(self, iterator, transforms, drop_untransformed=False, **kwargs):
        super(TransformingIterator, self).__init__(iterator=iterator, **kwargs)
        if drop_untransformed:
            self.OutputClass = AbstractDataIterator.create_output_class(transforms.keys())

        #these will get changed, the rest souldn't
        self.properties.pop('sources', None)
        
        self.transforms = transforms
        self.next_offset = 0

        if type(iterator) == TransformingIterator:
            self.iterator = iterator.iterator
            self_trans = self.transforms
            self.transforms = {}
            other_trans = iterator.transforms
            for s in self.source_names:
                of = other_trans.get(s, None)
                sf = self_trans.get(s, None)
                if of is None:
                    if sf is not None:
                        self.transforms[s] = sf
                else:
                    if sf is None:
                        self.transforms[s] = of
                    else:
                        self.transforms[s] = lambda X, of=of, sf=sf: sf(of(X)) 
        
    def next(self, peek=False):
        t = self.transforms
        utt = self.iterator.next(peek)
        return {s: t[s](utt[s]) if s in t else utt[s] for s in self.source_names}

    def reset(self):
        self.iterator.reset()

    def start(self, start_offset=0):
        self.iterator.start(start_offset)


class LimitBatchSizeIterator(AbstractWrappedIterator):
    def __init__(self, iterator, batch_size):
        super(LimitBatchSizeIterator, self).__init__(iterator=iterator)
        self.properties['batch_size'] = batch_size
        #todo: fix
        self.properties.pop('num_batches', None)
        self.batch_queue = []
        
    def next(self):
        if not self.batch_queue:
            utt = self.iterator.next()
            cutpoints = range(self.batch_size, utt[0].shape[0], self.batch_size)
            utt_split = [np.split(u, cutpoints) for u in utt]
            m = self.OutputClass._make
            self.batch_queue = [m(u) for u in izip(*utt_split)]
            self.batch_queue.reverse()
        return self.batch_queue.pop()


class _BatchIt:
    "A class to hold an utterance along with the index of the currrent row"
    
    __slots__ = ['batch','pos','len']
    
    def __init__(self, batch, permutation):
        self.batch=batch
        self.pos=0
        self.len=batch[0].shape[0]
        self.permutation = permutation


class ShuffledExamplesIterator(AbstractWrappedIterator):
    _default_seed = (17, 2, 946)
    
    def __init__(self, iterator, batch_size, 
                 shuffling_mem=100e6, rng=_default_seed):
        super(ShuffledExamplesIterator, self).__init__(iterator=iterator)
        self.shuffling_mem = shuffling_mem
        self.properties['batch_size'] = batch_size
        self.properties.pop('num_batches', None)
        self.properties['stochastic'] = True
        self.batch_pool = None
        self.mem_used = 0
        self.rng = make_np_rng(rng, which_method='random_integers')
        
    def _fill_to_mem_limit(self):
        for batch in self.iterator:
            self.mem_used += sum(u.nbytes for u in batch)
            permutation = self.rng.permutation(batch[0].shape[0]) 
            self.batch_pool.append(_BatchIt(batch, permutation))
            if self.mem_used >= self.shuffling_mem:
                break
        
    def next(self):
        #delay pool filling so that iterator creation is fast
        if self.batch_pool is None:
            self.batch_pool = []
            self._fill_to_mem_limit()
        batch_pool = self.batch_pool
        if not batch_pool:
            raise StopIteration
        
        #positions = self.rng.choice(len(batch_pool), self.batch_size)
        positions = self.rng.randint(len(batch_pool), size=self.batch_size)
        
        ret = self.make(np.empty((self.batch_size, batch.shape[1]), dtype=batch.dtype) 
                    for batch in batch_pool[0].batch)
        
        #fill row by row
        i = 0
        while i < positions.shape[0]:
            p = positions[i]
            batch = batch_pool[p]
            if batch.pos < batch.len:
                batchelem = batch.permutation[batch.pos]
                for r,u in izip(ret,batch.batch):
                    r[i,...] = u[batchelem,...]
                batch.pos += 1
                i += 1
            else: #utt.pos >= utt.len
                # swap with last unless we are the last
                last = batch_pool.pop()
                if last!=batch:
                    batch_pool[p] = last
                self.mem_used -= sum(u.nbytes for u in batch.batch)
                
                #fetch new utterances
                self._fill_to_mem_limit()
                
                if len(batch_pool)==0:
                    #return what we have right now...
                    return self.make(r[:i,...] for r in ret)
                
                rem_pos = positions[i:] #take a view
                bad_idx = rem_pos >= len(batch_pool)
                #rem_pos[bad_idx] = self.rng.choice(len(batch_pool), bad_idx.sum())
                rem_pos[bad_idx] = self.rng.randint(len(batch_pool), size=bad_idx.sum())
        return ret


class DataSpaceConformingIterator(AbstractWrappedIterator):
    def __init__(self, iterator, destination_data_specs, iterator_data_specs=None, rename_map=None, **kwargs):
        super(DataSpaceConformingIterator, self).__init__(iterator=iterator, **kwargs)
        
        if iterator_data_specs is None:
            iterator_data_specs = self.iterator.sources
        
        if rename_map is not None:
            self.OutputClass = AbstractDataIterator.create_output_class(destination_data_specs.keys())
        else:
            rename_map = dict((s,s) for s in self.source_names)
        
        iterator_source_names = iterator.source_names
        
        self._source_idxs = dict((sn, iterator_source_names.index(rename_map[sn])) for sn in self.source_names)
        self._conversion_funcs = {}
        
        for sn in self.source_names:
            iterator_space = iterator_data_specs[rename_map[sn]]
            destination_space = destination_data_specs[sn]

            def cf(X, iter_sp=iterator_space, dest_sp=destination_space):
                return iter_sp.np_format_as(X, dest_sp)
            self._conversion_funcs[sn] = cf
        
        self.properties['sources'] = destination_data_specs
    
    def next(self):
        batch = self.iterator.next()
        cf = self._conversion_funcs
        src_idx = self._source_idxs
        ret = self.make(cf[sn](batch[src_idx[sn]]) for sn in self.source_names)
        return ret


class BatchIterator(AbstractWrappedIterator):
    def __init__(self, iterator, big_batch_size=10000, mini_batch_size=1000, **kwargs):
        super(BatchIterator, self).__init__(iterator=iterator, **kwargs)

        self.big_batch_size = big_batch_size

        self.mini_batch_size = mini_batch_size
        self.next_offset = 0
        self.batch_position = 0
        self.current_big_batch = None

    def reset(self):
        self.batch_position = 0
        self.iterator.reset()

    def start(self, start_offset=0):
        self.batch_position = 0
        self.iterator.start(start_offset)

    def next(self, peek=False):
        def extend_with_zeros(arrays):
            max_length = max([len(arr) for arr in arrays])
            new_arrays = [np.append(arr, np.zeros(max_length - len(arr), dtype=arr.dtype)) for arr in arrays]
            masks = [np.append(np.ones(len(arr), dtype="float32"),
                               np.zeros(max_length - len(arr), dtype="float32"), axis=0) for arr in arrays]
            return new_arrays, masks

        def merge_batch(batch):
            keys = self.iterator.source_names
            out_dict = {}
            for key in keys:
                arrays = [d[key].T for d in batch]
                arrays, masks = extend_with_zeros(arrays)
                out_dict[key] = np.array(arrays).T
                out_dict[key + '_mask'] = np.array(masks).T
            return out_dict

        if self.current_big_batch and self.batch_position < self.big_batch_size:
            m_batch = self.current_big_batch[self.batch_position: self.batch_position + self.mini_batch_size]
            if not peek:
                self.batch_position += self.mini_batch_size
            new_batch = merge_batch(m_batch)
            return new_batch
        self.batch_position = 0
        batch = []
        for i in xrange(self.big_batch_size):
            batch += [self.iterator.next()]
        np.random.shuffle(batch)
        self.current_big_batch = batch
        mini_batch = batch[0:self.mini_batch_size]
        if not peek:
            self.batch_position += self.mini_batch_size
        ans = merge_batch(mini_batch)
        return ans


class OneExampleIterator(AbstractWrappedIterator):
    def __init__(self, iterator, **kwargs):
        super(OneExampleIterator, self).__init__(iterator=iterator, **kwargs)

        self.next_offset = 0

    def reset(self):
        self.iterator.reset()

    def start(self, start_offset=0):
        self.iterator.start(start_offset)

    def next(self, peek=False):
        def extend_with_zeros(arrays):
            max_length = max([len(arr) for arr in arrays])
            new_arrays = [np.append(arr, np.zeros(max_length - len(arr), dtype=arr.dtype)) for arr in arrays]
            masks = [np.append(np.ones(len(arr), dtype="float32"),
                               np.zeros(max_length - len(arr), dtype="float32"), axis=0) for arr in arrays]
            return new_arrays, masks

        def merge_batch(batch):
            keys = self.iterator.source_names
            out_dict = {}
            for key in keys:
                arrays = [d[key].T for d in batch]
                arrays, masks = extend_with_zeros(arrays)
                out_dict[key] = np.array(arrays).T
                out_dict[key + '_mask'] = np.array(masks).T
            return out_dict

        batch = [self.iterator.next(peek)]
        ans = merge_batch(batch)
        return ans


class CMUIterator(AbstractDataIterator):
    '''
    Iterates over CMU Pronunciation dictionary.
    An input file should be a pickle file with
    dictionary contains fields:
        train_words: a list of numpy arrays with indexes of words
        valid_words: the same as train_words
        test_words: the same as train_words
        train_phones: a list fo numpy arrays with indexes of phonemes
        valid_phones: the same as train_phones
        test_phones: the same as train_phones
        phone_vocab_size: int, size of phoneme vocabulary
        phone_vocabulary: dict, phone:index
        alphabet_size: int, size of alphabet
        alphabet: dict, character:index
    '''
    def __init__(self, filename, sources=('x', 'y'), subset='train'):
        """
            Read the CMU dictionary
        """
        super(CMUIterator, self).__init__(source_names=sources)
        self.subset = subset

        with open(filename, 'rt') as finp:
            data_dict = cPickle.load(finp)
            self.data_words = data_dict[subset + '_words']
            self.data_phones = data_dict[subset + '_phones']

        self.position = 0
        self.next_offset = 0
        self.size = len(self.data_words)

    def next(self, peek=False):
        if self.position >= self.size:
            raise StopIteration()
        utt_name, utt_feats = '', self.data_phones[self.position]
        utt_targets = self.data_words[self.position]
        if not peek:
            self.position += 1
        return dict(x=utt_feats, y=utt_targets)

    def start(self, start_offset=0):
        self.position = 0

    def reset(self):
        self.position = 0


class InfiniteIterator(AbstractWrappedIterator):
    def __init__(self, iterator):
        super(InfiniteIterator, self).__init__(iterator=iterator)

    def next(self, peek=False):
        try:
            return self.iterator.next(peek)
        except StopIteration:
            self.iterator.position = 0

    def start(self, start_offset=0):
        self.iterator.start(start_offset)

    def reset(self):
        self.iterator.reset()


def get_cmu_batch_iterator(subset, state, rng, logger, single_utterances=False, shuffle_override=None,
                           add_utterance_names=False, peek=False):
    if 'randomize_iterator' in state and state['randomize_iterator']:
        logger.info("Randomly resetting the random seed for data iterator")
        rng = np.random.RandomState()

    sources = ('x', 'y')

    def tfun_targets(feats):
        print "Targets ", feats.shape
        return np.append(feats, np.zeros((1, feats.shape[1]), dtype=feats.dtype), axis=0)

    def tfun_feats(feats):
        print "Features ", feats.shape
        return np.append(np.zeros((1, feats.shape[1]), dtype=feats.dtype), feats, axis=0)

    def tfun_mask_targets(feats):
        print "Mask features ", feats.shape
        return np.append(feats, np.ones(1, dtype=feats.dtype), axis=0)

    def tfun_mask_feats(feats):
        print "Mask targets ", feats.shape
        return np.append(np.ones(1, dtype=feats.dtype), feats, axis=0)

    def get_iter_fun(rng):
        sequence_iterator = CMUIterator(
            filename='/data/lisatmp3/serdyuk/cmudict/all_data.pkl',
            sources=sources,
            subset=subset
        )
        if subset == 'valid':
            #sequence_iterator = OneExampleIterator(sequence_iterator)
            return sequence_iterator
        sequence_iterator = InfiniteIterator(sequence_iterator)
        sequence_iterator = BatchIterator(sequence_iterator, big_batch_size=state['big_batch'],
                                          mini_batch_size=state['mini_batch'])
        #trans_seq_iter = TransformingIterator(sequence_iterator, dict(y=tfun_targets, x=tfun_feats,
        #                                                              y_mask=tfun_mask_targets, x_mask=tfun_mask_feats))
        return sequence_iterator

    return get_iter_fun(rng)
