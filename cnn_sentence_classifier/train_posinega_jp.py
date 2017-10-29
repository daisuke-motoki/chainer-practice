import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from chainer import serializers, iterators, optimizers, training
from chainer.training import extensions
from nets import CNNSentenceClassifier
from utils import make_vectorizer, text_to_word_sequence


def get_inputs(texts, labels, tokenizer, vocabulary, label_encoder,
               null_id=0, unknown_id=1):
    sequences = text_to_word_sequence(
        texts, tokenizer, vocabulary, null_id, unknown_id
    )
    
    gts = label_encoder.transform(labels).astype("int32")
    inputs = list()
    for sequence, gt in zip(sequences, gts):
        inputs.append((sequence, gt))

    return inputs


if __name__ == "__main__":
    # settings
    result_dir = "result/posinega_jp"
    null_id = 0
    unknown_id = 1
    labels = [1.0, 2.0, 3.0, 4.0, 5.0]
    option = "-d /Users/daisuke.motoki/local/lib/mecab/dic/mecab-ipadic-neologd"

    # load inputs
    headers = ["resource", "rating", "content"]
    filename = "../../../../data/posi_nega_corpus/review_data.csv"
    df = pd.read_csv(filename, names=headers)
    df = df[:1000]

    # vectorizer = make_vectorizer(df.content.tolist(),
    #                              option=option)
    # tokenizer = vectorizer.tokenizer
    # vocabulary = vectorizer.vocabulary_
    # init_E = None

    import word2vec
    word2vec_file = "../../../../data/wiki_word2vec_jp/entity_vector/entity_vector.model.txt"
    w2v = word2vec.load(word2vec_file)
    vocabulary = {"": 0, "###": 1}
    last_index = 2
    n_dim = len(w2v.vectors[0])
    init_E = np.zeros([2, n_dim])
    indexes = list()
    for i, word in enumerate(w2v.vocab):
        if word[0] == "[":
            continue
        vocabulary[word] = last_index
        last_index += 1
        indexes.append(i)
    init_E = np.concatenate((init_E, w2v.vectors[indexes]))

    from jp_extractor import JapaneseMecabWordExtractor
    tokenizer = JapaneseMecabWordExtractor(split_mode="unigram",
                                           use_all=True,
                                           tagger_option=option)

    label_encoder = LabelEncoder().fit(labels)

    # separate train and test
    indexes = np.arange(len(df))
    np.random.seed(0)
    np.random.shuffle(indexes)
    train_last = int(len(df)*0.8)
    train_indexes = indexes[:train_last]
    val_indexes = indexes[train_last:]

    # make inputs
    train = get_inputs(df.content[train_indexes].tolist(),
                       df.rating[train_indexes].tolist(),
                       tokenizer,
                       vocabulary,
                       label_encoder,
                       null_id,
                       unknown_id)
    test = get_inputs(df.content[val_indexes].tolist(),
                      df.rating[val_indexes].tolist(),
                      tokenizer,
                      vocabulary,
                      label_encoder,
                      null_id,
                      unknown_id)

    train_iter = iterators.SerialIterator(train,
                                          batch_size=1,
                                          shuffle=True)
    test_iter = iterators.SerialIterator(test,
                                         batch_size=1,
                                         shuffle=False,
                                         repeat=False)

    n_vocab = len(vocabulary.keys())
    n_units = 200
    model = CNNSentenceClassifier(n_vocab,
                                  n_units,
                                  filter_sizes=[1],
                                  n_filter=10,
                                  drop_rate=0.2,
                                  n_class=len(labels),
                                  init_E=init_E)
    model.embed.disable_update()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter,
                                       optimizer,
                                       loss_func=model.loss)
    trainer = training.Trainer(updater,
                               stop_trigger=(10, "epoch"),
                               out=result_dir)
    trainer.extend(extensions.Evaluator(test_iter, model,
                                        eval_func=model.loss))
    trainer.extend(extensions.dump_graph("main/loss"))
    trainer.extend(
        extensions.snapshot(filename="snapshot_{.updater.epoch}.npz"),
        trigger=(1, "epoch")
    )
    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ["main/loss", "validation/main/loss"],
                "epoch", file_name="loss.png"
            )
        )
    trainer.extend(
        extensions.PrintReport(
            ["epoch", "main/loss", "validation/main/loss", "elapsed_time"]
        )
    )
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    model.save_structure(result_dir + "/model_structure.json")
    serializers.save_npz(result_dir + "/model_weights.npz", model)
