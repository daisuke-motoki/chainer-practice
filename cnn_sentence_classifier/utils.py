import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from jp_extractor import JapaneseMecabWordExtractor
from chainer import iterators


def make_vectorizer(texts, null_set=(0, ""), unknown_set=(1, "###"),
                    option=None):
    """
    """
    print("Making vectorizer.")
    vectorizer = CountVectorizer(max_df=1.0,
                                 min_df=10,
                                 max_features=10000,
                                 stop_words=[null_set[1], unknown_set[1]])
    vectorizer.tokenizer = JapaneseMecabWordExtractor(split_mode="unigram",
                                                      use_all=True,
                                                      tagger_option=option)
    vectorizer.fit(texts)
    max_id = max(vectorizer.vocabulary_.values())
    prev_char = vectorizer.get_feature_names()[null_set[0]]
    vectorizer.vocabulary_[null_set[1]] = null_set[0]
    vectorizer.vocabulary_[prev_char] = max_id + 1
    prev_char = vectorizer.get_feature_names()[unknown_set[0]]
    vectorizer.vocabulary_[unknown_set[1]] = unknown_set[0]
    vectorizer.vocabulary_[prev_char] = max_id + 2
    return vectorizer


def text_to_word_sequence(texts, tokenizer, vocabulary,
                          null_id, unknown_id, max_length=None):
    """
    """
    print("Converting text to index sequences")
    num_unknown = 0
    total_words = 0
    sequences = list()
    for text in texts:
        words = tokenizer(text)
        maxl = max_length if max_length else len(words)
        ids = np.ones(maxl, dtype="int32")*null_id
        offset = 0 if len(words) > maxl else maxl - len(words)
        for i, word in enumerate(words):
            if i >= maxl:
                break
            ids[i+offset] = vocabulary.get(word, unknown_id)
            total_words += 1
            if ids[i+offset] == unknown_id:
                num_unknown += 1
        sequences.append(ids)

    print("unknown/words : {}/{} = {:.4f}".format(
        num_unknown, total_words, num_unknown/total_words))
    print("unknown/sentence : {}/{} = {:.4f}".format(
        num_unknown, len(sequences), num_unknown/len(sequences)))

    return sequences


class SerialPaddingIterator(iterators.SerialIterator):
    """
    """
    def __next__(self):
        """
        """
        batch = super(SerialPaddingIterator, self).__next__()
        sequences = [batch[i][0] for i in range(len(batch))]
        sequences = pad_sequences(sequences)
        for i, sequence in enumerate(sequences):
            batch[i] = (np.expand_dims(sequence, 0), batch[i][1])
        return batch

    next = __next__


### keras function ###
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    # Raises
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
