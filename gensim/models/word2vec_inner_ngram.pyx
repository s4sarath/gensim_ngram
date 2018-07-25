#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset

from cython.parallel import prange

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

from word2vec_inner cimport bisect_left, random_int32, \
     scopy, saxpy, sdot, dsdot, snrm2, sscal, \
     REAL_t, EXP_TABLE, \
     our_dot, our_saxpy, \
     our_dot_double, our_dot_float, our_dot_noblas, our_saxpy_noblas

from utils_any2vec import _compute_ngrams_words

REAL = np.float32

DEF MAX_SENTENCE_LEN = 10000
DEF MAX_SUBWORDS = 1000

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0

cdef unsigned long long fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1neg, const int size,
    const np.uint32_t word_index, const np.uint32_t *subwords_index, const np.uint32_t subwords_len,
    const REAL_t alpha, REAL_t *work, REAL_t *l1, unsigned long long next_random, REAL_t *word_locks_vocab,
    REAL_t *word_locks_ngrams) nogil:

    cdef long long a
    cdef np.uint32_t word2_index = subwords_index[0]
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(l1, 0, size * cython.sizeof(REAL_t))

    scopy(&size, &syn0_vocab[row1], &ONE, l1, &ONE)
    for d in range(1, subwords_len):
        our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_index[d] * size], &ONE, l1, &ONE)
    cdef REAL_t norm_factor = ONEF / subwords_len
    sscal(&size, &norm_factor, l1 , &ONE)

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, l1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha
        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, l1, &ONE, &syn1neg[row2], &ONE)
    our_saxpy(&size, &word_locks_vocab[word2_index], work, &ONE, &syn0_vocab[row1], &ONE)
    for d in range(1, subwords_len):
        our_saxpy(&size, &word_locks_ngrams[subwords_index[d]], work, &ONE, &syn0_ngrams[subwords_index[d]*size], &ONE)
    return next_random


cdef void fast_sentence_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1, const int size,
    const np.uint32_t *subwords_index, const np.uint32_t subwords_len, 
    const REAL_t alpha, REAL_t *work, REAL_t *l1, REAL_t *word_locks_vocab,
    REAL_t *word_locks_ngrams) nogil:

    cdef long long a, b
    cdef np.uint32_t word2_index = subwords_index[0]
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f, g, f_dot, lprob

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(l1, 0, size * cython.sizeof(REAL_t))

    scopy(&size, &syn0_vocab[row1], &ONE, l1, &ONE)
    for d in range(1, subwords_len):
        our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_index[d] * size], &ONE, l1, &ONE)
    cdef REAL_t norm_factor = ONEF / subwords_len
    sscal(&size, &norm_factor, l1 , &ONE)

    for b in range(codelen):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, l1, &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, l1, &ONE, &syn1[row2], &ONE)

    our_saxpy(&size, &word_locks_vocab[word2_index], work, &ONE, &syn0_vocab[row1], &ONE)
    for d in range(1, subwords_len):
        our_saxpy(&size, &word_locks_ngrams[subwords_index[d]], work, &ONE, &syn0_ngrams[subwords_index[d]*size], &ONE)


cdef unsigned long long fast_sentence_cbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const np.uint32_t *subwords_idx[MAX_SENTENCE_LEN],
    const int subwords_idx_len[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, REAL_t *word_locks_vocab, REAL_t *word_locks_ngrams) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, log_e_f_dot, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        count += ONEF
        our_saxpy(&size, &ONEF, &syn0_vocab[indexes[m] * size], &ONE, neu1, &ONE)
        for d in range(subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_idx[m][d] * size], &ONE, neu1, &ONE)

    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)

    for m in range(j,k):
        if m == i:
            continue
        our_saxpy(&size, &word_locks_vocab[indexes[m]], work, &ONE, &syn0_vocab[indexes[m]*size], &ONE)
        for d in range(subwords_idx_len[m]):
            our_saxpy(&size, &word_locks_ngrams[subwords_idx[m][d]], work, &ONE, &syn0_ngrams[subwords_idx[m][d]*size], &ONE)

    return next_random


cdef void fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const np.uint32_t *subwords_idx[MAX_SENTENCE_LEN],
    const int subwords_idx_len[MAX_SENTENCE_LEN],const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *word_locks_vocab, REAL_t *word_locks_ngrams) nogil:

    cdef long long a, b
    cdef long long row2, sgn
    cdef REAL_t f, g, count, inv_count = 1.0, f_dot, lprob
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        count += ONEF
        our_saxpy(&size, &ONEF, &syn0_vocab[indexes[m] * size], &ONE, neu1, &ONE)
        for d in range(subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_idx[m][d] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)

    for m in range(j,k):
        if m == i:
            continue
        our_saxpy(&size, &word_locks_vocab[indexes[m]], work, &ONE, &syn0_vocab[indexes[m]*size], &ONE)
        for d in range(subwords_idx_len[m]):
            our_saxpy(&size, &word_locks_ngrams[subwords_idx[m][d]], work, &ONE, &syn0_ngrams[subwords_idx[m][d]*size], &ONE)

cdef unsigned long long fast_sentence_sg_neg_w2v(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0, REAL_t *syn1neg, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
    unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_dot, log_e_f_dot
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1neg[row2], &ONE)

    our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)

    return next_random


cdef unsigned long long fast_sentence_cbow_neg_ngram(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0_vocab, REAL_t *syn0_ngrams, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const np.uint32_t *subwords_idx[MAX_SENTENCE_LEN],
    const int subwords_idx_len[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int cbow_mean, int indexlen, unsigned long long next_random, REAL_t *word_locks_vocab, REAL_t *word_locks_ngrams) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, log_e_f_dot, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m


    word_index = i

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(indexlen):

        if indexes[m] == i:
            continue
        count += ONEF
        our_saxpy(&size, &ONEF, &syn0_vocab[indexes[m] * size], &ONE, neu1, &ONE)
        for d in range(subwords_idx_len[m]):
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0_ngrams[subwords_idx[m][d] * size], &ONE, neu1, &ONE)

    if count > (<REAL_t>0.5):
        inv_count = ONEF / count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)

    for m in range(indexlen):
        if indexes[m] == i:
            continue
        our_saxpy(&size, &word_locks_vocab[indexes[m]], work, &ONE, &syn0_vocab[indexes[m]*size], &ONE)
        for d in range(subwords_idx_len[m]):
            our_saxpy(&size, &word_locks_ngrams[subwords_idx[m][d]], work, &ONE, &syn0_ngrams[subwords_idx[m][d]*size], &ONE)

    return next_random





cdef unsigned long long fast_sentence_cbow_neg_w2v(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int cbow_mean, int indexlen, unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_running_training_loss_param) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, log_e_f_dot, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = i

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(indexlen):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _running_training_loss_param[0] = _running_training_loss_param[0] - log_e_f_dot

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(indexlen):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    return next_random











def train_batch_sg_ngram(model, sentences, word_gram , context_gram, alpha, _work, compute_loss, keep_overlap=1):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int keep_overlap_here = keep_overlap

    cdef word_gram_min = word_gram[0]
    cdef word_gram_max = word_gram[1]
    cdef context_gram_min = context_gram[0]
    cdef context_gram_max = context_gram[1]

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    n_gram_vocab_temp = {}
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for index_, sentence_ngram in enumerate(_compute_ngrams_words(sent, max_n=max(word_gram_max, context_gram_max))):
            n_gram_vocab_temp[index_] = sentence_ngram
        for word_iter in range(word_gram_max):
            sent_temp = n_gram_vocab_temp[word_iter]
            for pos,token in enumerate(sent_temp):
                word = vlookup[token] if token in vlookup else None
                if word is None:
                    continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
                if sample and word.sample_int < random_int32(&next_random):
                    continue
                reduced_window = model.random.randint(model.window)
                start = max(0, pos - model.window + reduced_window)

                indexes[effective_words] = word.index
                
                for context_iter in range(context_gram_max):
                    context_window = n_gram_vocab_temp[context_iter]
                    shift_window   = pos + model.window - reduced_window + 1
                    current_window = context_window[start:(shift_window-context_iter)]
                    for pos2, word2 in enumerate(current_window, start):
                        if word2 not in vlookup:
                            continue
                        if word2 == token:
                            continue
                        if word2 not in vlookup:
                            continue
                        if pos2 == pos and word==word2:# don't train on the `word` itself
                            continue
                        if keep_overlap_here != 1 and context_iter > 0:
                            if word in word2:
                                if pos2 in range(pos-context_iter, pos+1):
                                    continue
                                    
                        if negative:
                            next_random = fast_sentence_sg_neg_w2v(negative, cum_table, cum_table_len, syn0, syn1neg, size,vlookup[word2].index, word.index, _alpha, work, next_random, word_locks, _compute_loss, &_running_training_loss)
                effective_words += 1
    
    model.running_training_loss = _running_training_loss
    return effective_words





def train_batch_cbow_ngram(model, sentences, word_gram, context_gram, alpha, _work, _neu1, compute_loss, keep_overlap=1):

    cdef int keep_overlap_here = keep_overlap

    cdef word_gram_min = word_gram[0]
    cdef word_gram_max = word_gram[1]
    cdef context_gram_min = context_gram[0]
    cdef context_gram_max = context_gram[1]

    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int cbow_mean = model.cbow_mean

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _running_training_loss = model.running_training_loss

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int counter = 0

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random
    n_gram_vocab_temp = {}
    

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for index_, sentence_ngram in enumerate(_compute_ngrams_words(sent, max_n=max(word_gram_max, context_gram_max))):
            n_gram_vocab_temp[index_] = sentence_ngram
        for word_iter in range(word_gram_max):
            sent_temp = n_gram_vocab_temp[word_iter]
            for pos,token in enumerate(sent_temp):
                        
                word = vlookup[token] if token in vlookup else None
                if word is None:
                    continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
                if sample and word.sample_int < random_int32(&next_random):
                    continue
                reduced_window = model.random.randint(model.window)
                start = max(0, pos - model.window + reduced_window)
                


                for context_iter in range(context_gram_max):
                    context_window = n_gram_vocab_temp[context_iter]
                    shift_window   = pos + model.window - reduced_window + 1
                    current_window = context_window[start:(shift_window-context_iter)]
                    

                    
                    for pos2, word2 in enumerate(current_window, start):
                        if word2 not in vlookup:
                            continue
                        if word2 == token:
                            continue
                        if pos2 == pos and word==word2:# don't train on the `word` itself
                            continue
                        if keep_overlap_here != 1 and context_iter > 0:
                            if word in word2:
                                if pos2 in range(pos-context_iter, pos+1):
                                    continue
                        indexes[counter] = vlookup[word2].index                        
                        counter += 1
                        
                if negative:
                    next_random = fast_sentence_cbow_neg_w2v(negative, cum_table, cum_table_len, codelens, neu1, syn0, syn1neg, size, indexes, _alpha, work, word.index, cbow_mean, counter, next_random, word_locks, _compute_loss, &_running_training_loss)
                effective_words += 1
                counter = 0
        
    model.running_training_loss = _running_training_loss
    return effective_words



def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.
    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        our_dot = our_dot_double
        our_saxpy = saxpy
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        our_dot = our_dot_float
        our_saxpy = saxpy
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN

