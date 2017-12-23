from chainer import Chain
from chainer import initializers
from chainer import report
import chainer.functions as F
import chainer.links as L
from collections import OrderedDict
from links.connection.caps_conv_2d import CapsConv2D
from links.connection.caps_linear import CapsLinear
from functions.activation.squash import squash


class CapsNet(Chain):
    """
    """
    def __init__(self, recon_loss_weight=0.0005):
        """
        """
        super(CapsNet, self).__init__()
        self.recon_loss_weight = recon_loss_weight

        init_W = initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 256, ksize=9, stride=1,
                                         nobias=True, initialW=init_W)

            self.primarycaps = CapsConv2D(256, 1, 32, 8, ksize=9, stride=2,
                                          nobias=True, initialW=init_W)

            self.digitcaps = CapsLinear(32*6*6, 8, 10, 16)

            # for reconstruction
            self.fc1 = L.Linear(16 * 10, 512, initialW=init_W)
            self.fc2 = L.Linear(512, 1024, initialW=init_W)
            self.fc3 = L.Linear(1024, 784, initialW=init_W)

    def __call__(self, x, layers=["prob"], **kwargs):
        """
        """
        activations = dict()

        # inputs
        act = x

        # conv1
        act = F.relu(self.conv1(act))
        if "conv1" in layers:
            activations["conv1"] = act

        # primary capsule
        hidden = self.primarycaps(act)
        act = squash(hidden, axis=2)
        if "primarycaps" in layers:
            activations["primarycaps"] = act

        # digit capsule
        hidden = self.digitcaps(act)
        if "digitcaps" in layers:
            activations["digitcaps"] = hidden

        act = self._length(hidden)
        if "prob" in layers:
            activations["prob"] = act

        return activations

    def _length(self, x):
        """
        """
        return F.sqrt(F.sum(F.square(x), -1))

    def _margin_loss(self, x, t, m_plus=0.9, m_minus=0.1, l=0.5):
        """
        """
        batchsize = x.shape[0]
        t_discreted = self.xp.zeros(x.shape, dtype="float32")
        t_discreted[self.xp.arange(batchsize), t] = 1.
        L_present = t_discreted * F.square(F.relu(m_plus - x))
        L_absent = (1. - t_discreted) * F.square(F.relu(x - m_minus))
        L = L_present + l * L_absent
        return F.sum(L) / float(batchsize)

    def reconstruct(self, x, t):
        """
        """
        batchsize = x.shape[0]
        t_discreted = self.xp.zeros(x.shape, dtype="float32")
        t_discreted[self.xp.arange(batchsize), t] = 1.
        masked_x = t_discreted * x

        act = F.relu(self.fc1(masked_x))
        act = F.relu(self.fc2(act))
        act = F.sigmoid(self.fc3(act))
        return act

    def loss(self, x, t, **kwargs):
        """
        """
        activations = self.__call__(x, layers=["digitcaps", "prob"])
        reconstructions = self.reconstruct(activations["digitcaps"], t)

        # classification loss
        c_loss = self._margin_loss(activations["prob"], t)
        # reconstruction loss
        r_loss = F.mean_squared_error(x, reconstructions.reshape(x.shape))

        total_loss = c_loss + self.recon_loss_weight * r_loss
        report({"loss": total_loss, "c_loss": c_loss, "r_loss": r_loss}, self)

        return total_loss

    def extract(self):
        """
        """
        pass

    def predict(self):
        """
        """
        pass
