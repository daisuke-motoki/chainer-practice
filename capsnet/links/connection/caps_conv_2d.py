from chainer import link
import chainer.links as L
import chainer.functions as F


class CapsConv2D(link.Link):
    """
    """
    def __init__(self, in_channels, in_dims, out_channels, out_dims, **kwargs):
        """
        """
        super(CapsConv2D, self).__init__()

        self.in_channels = in_channels
        self.in_dims = in_dims
        self.out_channels = out_channels
        self.out_dims = out_dims

        with self.init_scope():
            self.capsules = L.Convolution2D(in_channels*in_dims,
                                            out_channels*out_dims,
                                            **kwargs)

    def __call__(self, x):
        """
        """
        hidden = self.capsules(x)
        out_shape = (-1, self.out_channels, self.out_dims,
                     hidden.shape[-2], hidden.shape[-1])
        return F.reshape(hidden, out_shape)
