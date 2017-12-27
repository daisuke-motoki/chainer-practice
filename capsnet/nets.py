import chainer
from chainer import function
from chainer import Chain
from chainer import initializers
from chainer import report
from chainer.variable import Variable
from chainer.dataset.convert import concat_examples
import chainer.functions as F
import chainer.links as L
from links.connection.caps_conv import CapsConv
from links.connection.caps_linear import CapsLinear


class CapsNet(Chain):
    """
    """
    def __init__(self, grid_weight_share=False):
        """
        """
        super(CapsNet, self).__init__()
        self.grid_weight_share = grid_weight_share
        self.recon_loss_weight = 0.0005
        init_scale = 0.1

        # init_W = initializers.HeNormal()
        init_W = initializers.Uniform(scale=init_scale)
        conv1_param = dict(in_channels=1, out_channels=256,
                           ksize=9, stride=1,
                           nobias=True, initialW=init_W)
        pcaps_param = dict(in_caps=1, in_dims=256,
                           out_caps=32, out_dims=8,
                           n_iters=1,
                           ksize=9, stride=2,
                           nobias=True, initialW=init_W)
        if self.grid_weight_share:
            dcapsconv_param = dict(in_caps=32, in_dims=8,
                                   out_caps=10, out_dims=16,
                                   n_iters=3, flat_output=True,
                                   ksize=1, stride=1,
                                   nobias=True, initialW=init_W)
        else:
            dcapslin_param = dict(in_caps=32*6*6, in_dims=8,
                                  out_caps=10, out_dims=16,
                                  n_iters=3,
                                  initialW=init_W)
        with self.init_scope():
            self.conv1 = L.Convolution2D(**conv1_param)
            self.primarycaps = CapsConv(**pcaps_param)

            if self.grid_weight_share:
                self.digitcaps = CapsConv(**dcapsconv_param)
            else:
                self.digitcaps = CapsLinear(**dcapslin_param)

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
        act = self.primarycaps(act[:, None])
        if "primarycaps" in layers:
            activations["primarycaps"] = act

        # digit capsule
        act = self.digitcaps(act)
        if "digitcaps" in layers:
            activations["digitcaps"] = act

        act = self._length(act)
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
        return F.mean(F.sum(L, axis=-1))

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
        activations = self.extract(x, layers=["digitcaps", "prob"])
        recons = self.reconstruct(activations["digitcaps"], t)

        # classification loss
        c_loss = self._margin_loss(activations["prob"], t)
        # reconstruction loss
        recons = recons.reshape(x.shape)
        r_loss = F.mean(F.sum(F.squared_error(x, recons), axis=(1, 2, 3)))
        r_loss *= self.recon_loss_weight

        total_loss = c_loss + r_loss
        report({"loss": total_loss, "c_loss": c_loss, "r_loss": r_loss}, self)
        accuracy = F.accuracy(activations["prob"], t)
        report({"accuracy": accuracy}, self)

        return total_loss

    def extract(self, images, layers=["digitcaps"]):
        """
        """
        x = concat_examples([preprocess(image) for image in images])
        x = Variable(self.xp.asarray(x))
        activations = self(x, layers=layers)
        return activations

    def predict(self, images, layers=["prob"]):
        """
        """
        with function.no_backprop_mode(), chainer.using_config("train", False):
            activations = self.extract(images, layers=layers)
        return activations


def preprocess(image):
    """
    """
    return image
