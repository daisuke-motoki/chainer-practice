import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from jp_extractor import JapaneseMecabWordExtractor


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
        sequences.append(np.array([ids]))

    print("unknown/words : {}/{} = {:.4f}".format(
        num_unknown, total_words, num_unknown/total_words))
    print("unknown/sentence : {}/{} = {:.4f}".format(
        num_unknown, len(sequences), num_unknown/len(sequences)))

    return sequences
