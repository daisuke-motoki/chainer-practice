import collections
from chainer.links import VGG16Layers
from chainer.functions.pooling.max_pooling_2d import max_pooling_2d
from chainer.functions.activation.relu import ReLU
from chainer.functions.noise.dropout import dropout
from chainer.functions.activation.softmax import softmax


class GuidedReLU(ReLU):
    """
    """
    def backward(self, indexes, gy):
        """
        """
        y = self.get_retained_outputs()[0].data
        f_l = (y > 0).astype(y.dtype)
        R_lp = (gy[0].data > 0)
        return f_l * R_lp * gy[0],


class GuidedVGG16(VGG16Layers):
    """
    """
    def __call__(self, x, layers=["prob"], **kwargs):
        """
        """
        activations = {}
        if "input" in layers:
            activations["input"] = x

        acts = super(GuidedVGG16, self).__call__(x, layers, **kwargs)

        for key, value in acts.items():
            activations[key] = value

        return activations

    @property
    def functions(self):
        relu = guided_relu
        return collections.OrderedDict([
            ('conv1_1', [self.conv1_1, relu]),
            ('conv1_2', [self.conv1_2, relu]),
            ('pool1', [_max_pooling_2d]),
            ('conv2_1', [self.conv2_1, relu]),
            ('conv2_2', [self.conv2_2, relu]),
            ('pool2', [_max_pooling_2d]),
            ('conv3_1', [self.conv3_1, relu]),
            ('conv3_2', [self.conv3_2, relu]),
            ('conv3_3', [self.conv3_3, relu]),
            ('pool3', [_max_pooling_2d]),
            ('conv4_1', [self.conv4_1, relu]),
            ('conv4_2', [self.conv4_2, relu]),
            ('conv4_3', [self.conv4_3, relu]),
            ('pool4', [_max_pooling_2d]),
            ('conv5_1', [self.conv5_1, relu]),
            ('conv5_2', [self.conv5_2, relu]),
            ('conv5_3', [self.conv5_3, relu]),
            ('pool5', [_max_pooling_2d]),
            ('fc6', [self.fc6, relu, dropout]),
            ('fc7', [self.fc7, relu, dropout]),
            ('fc8', [self.fc8]),
            ('prob', [softmax]),
        ])


def _max_pooling_2d(x):
    return max_pooling_2d(x, ksize=2)


def guided_relu(x):
    y, = GuidedReLU().apply((x,))
    return y
