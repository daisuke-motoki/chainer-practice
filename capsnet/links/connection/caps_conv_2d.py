import chainer.links as L
import chainer.functions as F


class CapsConv2D(L.Convolution2D):
    """
    """
    def __init__(self, in_caps, in_dims, out_caps, out_dims, **kwargs):
        """
        """
        super(CapsConv2D, self).__init__(in_caps*in_dims,
                                         out_caps*out_dims,
                                         **kwargs)

        self.in_caps = in_caps
        self.in_dims = in_dims
        self.out_caps = out_caps
        self.out_dims = out_dims

    def __call__(self, x):
        """
        """
        hidden = super(CapsConv2D, self).__call__(x)
        out_shape = (-1, self.out_caps, self.out_dims,
                     hidden.shape[-2], hidden.shape[-1])
        return F.reshape(hidden, out_shape)
