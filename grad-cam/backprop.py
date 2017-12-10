import copy
import chainer
import chainer.functions as F
from scipy.misc import imresize


class Backprop:
    """
    """
    def __init__(self, model, target_layer="conv5_3", prob_layer="prob"):
        """
        """
        self.model = copy.deepcopy(model)
        self.xp = self.model.xp
        self.target_layer = target_layer
        self.prob_layer = prob_layer

    def forward(self, x):
        """
        """
        with chainer.using_config("train", False):
            layers = [self.target_layer, self.prob_layer]
            activations = self.model.extract(x, layers=layers)

        return activations

    def backward(self, target_prob):
        """
        """
        self.model.cleargrads()
        loss = F.sum(target_prob)
        loss.backward(retain_grad=True)


class GradCAM(Backprop):
    """
    """
    def feed(self, x, label):
        """ feed
        Args:
            x: list or array: Input image. Only one image can be acceptable.
            label: int: The number of class label.
        Return:
            L_gcam: Grad-CAM result.
        """
        # feed forward
        activations = self.forward(x)

        # label selection
        prob = activations[self.prob_layer][0].data
        n_class = len(prob)
        target_label = self.xp.zeros((1, n_class), dtype=self.xp.float32)
        if label == -1:
            target_label[0, prob.argmax()] = 1
        else:
            target_label[0, label] = 1

        # target loss
        target_prob = \
            chainer.Variable(target_label) * activations[self.prob_layer]

        # backward
        self.backward(target_prob)
        target_activation = activations[self.target_layer]
        importances = self.xp.mean(target_activation.grad, axis=(2, 3))
        L_gcam = self.xp.tensordot(importances[0],
                                   target_activation.data[0],
                                   axes=(0, 0))
        L_gcam = (L_gcam > 0) * L_gcam / L_gcam.max() * 255.

        # resize
        L_gcam = imresize(L_gcam, x[0].size)

        return L_gcam
