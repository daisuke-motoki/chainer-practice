import chainer
import chainer.functions as F
from scipy.misc import imresize


class Backprop:
    """ Backprop
    """
    def __init__(self, model, target_layer="conv5_3", prob_layer="prob"):
        """ init
        """
        self.model = model
        self.xp = self.model.xp
        self.target_layer = target_layer
        self.prob_layer = prob_layer

    def forward(self, x):
        """ forward
        """
        with chainer.using_config("train", False):
            layers = [self.target_layer, self.prob_layer]
            activations = self.model.extract(x, layers=layers)

        return activations

    def backward(self, target_prob):
        """ backward
        """
        self.model.cleargrads()
        loss = F.sum(target_prob)
        loss.backward(retain_grad=True)

    def select_target(self, prob, label):
        """
        """
        n_class = len(prob)
        target_label = self.xp.zeros((1, n_class), dtype=self.xp.float32)
        if label == -1:
            target_label[0, prob.argmax()] = 1.
        else:
            target_label[0, label] = 1.

        return target_label


class GradCAM(Backprop):
    """ Grad-CAM
    """
    def __init__(self, model, target_layer="conv5_3", prob_layer="prob"):
        """ init
        """
        super(GradCAM, self).__init__(model, target_layer, prob_layer)

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
        target_label = self.select_target(prob, label)

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
        L_gcam = (L_gcam > 0.) * L_gcam / L_gcam.max() * 255.

        # resize
        L_gcam = imresize(L_gcam, x[0].size)

        return L_gcam


class GradCAM_PP(Backprop):
    """ Grad-CAM++
    """
    def __init__(self, model, target_layer="conv5_3", prob_layer="prob"):
        """ init
        """
        super(GradCAM_PP, self).__init__(model, target_layer, prob_layer)

    def feed(self, x, label):
        """ feed
        Args:
            x: list or array: Input image. Only one image can be acceptable.
            label: int: The number of class label.
        Return:
            L_gcam: Grad-CAM++ result.
        """
        # feed forward
        activations = self.forward(x)

        # label selection
        prob = activations[self.prob_layer][0].data
        target_label = self.select_target(prob, label)

        # target loss
        target_prob = \
            chainer.Variable(target_label) * activations[self.prob_layer]

        # backward
        self.backward(target_prob)
        target_activation = activations[self.target_layer]
        label_index = target_label.argmax()
        first_grad = self.xp.exp(prob)[label_index] * target_activation.grad
        second_grad = first_grad * target_activation.grad
        third_grad = second_grad * target_activation.grad
        global_sum = self.xp.sum(target_activation.data, axis=(2, 3))
        global_sum = global_sum.reshape(first_grad[0].shape[0], 1, 1)
        alpha_num = second_grad[0]
        alpha_denom = 2.0 * second_grad[0] + global_sum[0] * third_grad[0]
        alpha_denom = self.xp.where(alpha_denom != 0.0,
                                    alpha_denom,
                                    self.xp.ones(alpha_denom.shape))
        alphas = alpha_num / alpha_denom
        # alphas /= self.xp.sum(alphas, axis=(1, 2))[:, self.xp.newaxis, self.xp.newaxis]
        importances = self.xp.sum(
            # alphas * self.xp.maximum(first_grad[0], 0.0),
            # # When I apply ReLU to gradients, the result is class
            # # discriminatve. This code might be wrong somewhere...
            alphas * first_grad[0],
            axis=(1, 2)
        )

        L_gcam = self.xp.tensordot(importances,
                                   target_activation.data[0],
                                   axes=(0, 0))
        L_gcam = (L_gcam > 0.) * L_gcam / L_gcam.max() * 255.

        # resize
        L_gcam = imresize(L_gcam, x[0].size)

        return L_gcam


class GuidedBackprop(Backprop):
    """ Guided Backpropagation
    """
    def __init__(self, model, target_layer="input", prob_layer="prob"):
        """ init
        """
        super(GuidedBackprop, self).__init__(model, target_layer, prob_layer)

    def feed(self, x, label):
        """ feed
        Args:
            x: list or array: Input image. Only one image can be acceptable.
            label: int: The number of class label.
        Return:
            L_gcam: Guided Backpropagation result.
        """
        # feed forward
        activations = self.forward(x)

        # label selection
        prob = activations[self.prob_layer][0].data
        target_label = self.select_target(prob, label)

        # target loss
        target_prob = \
            chainer.Variable(target_label) * activations[self.prob_layer]

        # backward
        self.backward(target_prob)
        L_gbp = activations[self.target_layer].grad[0]
        L_gbp = L_gbp.transpose(1, 2, 0)

        # resize
        # L_gbp = imresize(L_gbp, x[0].size)

        return L_gbp
