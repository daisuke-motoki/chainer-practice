from chainer import report
from chainer import Chain
from chainer import initializers
import chainer.functions as F
import chainer.links as L


class AutoEncoder(Chain):
    """
    """
    def __init__(self, input_dim, n_units):
        """ init
        Args:
            n_units: list, int: List of the number of units.
        """
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.n_units = n_units
        self.n_link = len(n_units)

        W = initializers.HeNormal()
        self.add_link("encode_0",
                      L.Linear(input_dim, n_units[0], initialW=W))
        for i in range(1, self.n_link, 1):
            self.add_link("encode_{}".format(i),
                          L.Linear(n_units[i-1], n_units[i], initialW=W))
        for i in range(self.n_link-1, 0, -1):
            self.add_link("decode_{}".format(i),
                          L.Linear(n_units[i], n_units[i-1], initialW=W))
        self.add_link("decode_0",
                      L.Linear(n_units[0], input_dim, initialW=W))

    def __call__(self, x):
        """
        """
        return self.decode(self.encode(x))

    def encode(self, x):
        """
        """
        for i in range(0, self.n_link, 1):
            x = F.relu(self["encode_{}".format(i)](x))
        return x

    def decode(self, x):
        """
        """
        for i in range(self.n_link-1, -1, -1):
            if i == 0:
                x = F.sigmoid(self["decode_{}".format(i)](x))
            else:
                x = F.relu(self["decode_{}".format(i)](x))
        return x

    def loss(self, x):
        """
        """
        y = self.decode(self.encode(x))
        loss_value = F.mean_squared_error(x, y)
        report({"loss": loss_value}, self)
        return loss_value

    def save_structure(self, filename):
        """
        """
        import json
        json.dump(
            dict(
                input_dim=self.input_dim,
                n_units=self.n_units
            ),
            open(filename, "w"),
            indent=4
        )
