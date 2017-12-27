import chainer.functions as F
from chainer import link
from chainer import initializers
from chainer import variable
from functions.connection.dynamic_routing import routing


class CapsLinear(link.Link):
    """
    """
    def __init__(self, in_caps, in_dims, out_caps, out_dims,
                 n_iters=3, initialW=None, caps_dim=2):
        """
        """
        super(CapsLinear, self).__init__()
        self.in_caps = in_caps
        self.in_dims = in_dims
        self.out_caps = out_caps
        self.out_dims = out_dims
        self.n_iters = n_iters
        self.caps_dim = caps_dim

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            self._initialize_params()
        self.b = None

    def _initialize_params(self):
        """
        """
        W_shape = (1, self.out_caps, self.in_caps,
                   self.out_dims, self.in_dims)
        self.W.initialize(W_shape)

    def __call__(self, x):
        """ call
        Args:
            x: [batch, n_global_capsule, caps_dim, n_local_grid, n_local_grid]
                ex) [?, 32, 8, 6, 6]
                    -> [?, 32, 6, 6, 8]
                    -> [?, 10, 1152, 8, 1]
        """
        # calculating x_hat
        x = F.swapaxes(x, self.caps_dim, -1)
        x = F.reshape(x, (-1, self.in_caps, self.in_dims))
        x = F.expand_dims(x, -1)
        x = F.expand_dims(x, 1)
        x = F.tile(x, (1, self.out_caps, 1, 1, 1))
        Ws = F.tile(self.W, (x.shape[0], 1, 1, 1, 1))
        x_hats = F.matmul(Ws, x)

        # dynamic routing
        x_hats = F.swapaxes(x_hats, 2, 3)
        x_hats = F.reshape(x_hats, x_hats.shape[:-1])
        v_j = routing(x_hats, self.n_iters)

        return v_j
