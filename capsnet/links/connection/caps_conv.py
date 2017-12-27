import chainer
import chainer.links as L
import chainer.functions as F
from functions.connection.dynamic_routing import routing


class CapsConv(chainer.ChainList):
    """
    """
    def __init__(self,
                 in_caps, in_dims,
                 out_caps, out_dims,
                 n_iters=1, flat_output=False,
                 **kwargs):
        """
        """
        super(CapsConv, self).__init__(
            *[L.Convolution2D(in_dims, out_dims*out_caps, **kwargs)
              for _ in range(in_caps)]
        )
        self.in_caps = in_caps
        self.in_dims = in_dims
        self.out_caps = out_caps
        self.out_dims = out_dims
        self.n_iters = n_iters
        self.flat_output = flat_output

    def __call__(self, x):
        """
        """
        self._check_shape(x.shape)
        n_batch = x.shape[0]
        x_hats = list()
        for i in range(self.in_caps):
            x_hat = self[i](x[:, i])
            g = x_hat.shape[-1]
            shape = (n_batch, self.out_caps, self.out_dims, g, g)
            x_hat = x_hat.reshape(shape)
            x_hats.append(x_hat)
        x_hats = F.stack(x_hats, axis=3)

        if self.n_iters > 0:
            if self.flat_output:
                x_hats = x_hats.reshape(
                    n_batch, self.out_caps, self.out_dims, -1)
            v_j = routing(x_hats, self.n_iters)
        else:
            g = x_hats.shape[-1]
            shape = (n_batch, self.out_caps, self.out_dims, g, g)
            v_j = x_hats.reshape(shape)

        return v_j

    def _check_shape(self, shape):
        """
        """
        if len(shape) != 5:
            raise ValueError()
        if shape[1] != self.in_caps:
            raise ValueError()
        if shape[2] != self.in_dims:
            raise ValueError()
