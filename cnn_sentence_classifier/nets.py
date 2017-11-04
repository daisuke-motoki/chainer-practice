import chainer
from chainer import Chain
from chainer import report
from chainer import initializers
import chainer.functions as F
import chainer.links as L


class CNNSentenceClassifier(Chain):
    """ Convolutional Neural Networks for Sentence Classification
    """
    def __init__(self, n_vocab, n_units, filter_sizes, n_filter,
                 drop_rate, n_class, init_E=None):
        """
        """
        self.n_vocab = n_vocab
        self.n_units = n_units
        self.filter_sizes = filter_sizes
        self.n_filter = n_filter
        self.drop_rate = drop_rate
        self.n_class = n_class
        super(CNNSentenceClassifier, self).__init__()

        with self.init_scope():
            # Embedding
            if init_E is None:
                init_E = initializers.Uniform(1. / n_units)
            self.add_link(
                "embed",
                L.EmbedID(n_vocab, n_units, initialW=init_E)
            )

            # Convolutions
            for sz in filter_sizes:
                self.add_link(
                    "conv_{}".format(sz),
                    L.Convolution2D(
                        1, n_filter, ksize=(sz, n_units), stride=1, pad=0,
                        initialW=initializers.HeNormal()
                    )
                )

            # FC
            self.add_link(
                "fc",
                L.Linear(
                    len(filter_sizes)*n_filter, n_class,
                    initialW=initializers.HeNormal()
                )
            )

    def __call__(self, x):
        """
        """
        return self.predict(x)

    def predict(self, x, train=False):
        """
        """
        x = self["embed"](x)
        nb, nf, nw, nd = x.shape
        hiddens = list()
        for sz in self.filter_sizes:
            hidden = F.relu(self["conv_{}".format(sz)](x))
            hidden = F.max_pooling_2d(hidden, nw+1-sz)
            hidden = F.reshape(hidden, (nb, self.n_filter))
            hiddens.append(hidden)
        h = F.concat(hiddens, axis=-1)
        # with chainer.using_config("train", train):
        h = F.dropout(h, ratio=self.drop_rate)

        if train:
            y = self["fc"](h)
        else:
            y = F.softmax(self["fc"](h))

        return y

    def loss(self, x, t):
        """
        """
        y = self.predict(x, train=True)
        loss_value = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({"loss": loss_value, "accuracy": accuracy}, self)

        return loss_value
