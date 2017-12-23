import chainer
import chainer.links as L
import chainer.functions as F
from chainer import link
from chainer import initializers
from chainer import variable
from functions.activation.squash import squash


class CapsLinear(link.Link):
    """
    """
    def __init__(self, in_caps, in_dims, out_caps, out_dims,
                 n_iters=3, initialW=None, caps_dim=2, learn_coupling=True):
        """
        """
        super(CapsLinear, self).__init__()
        self.in_caps = in_caps
        self.in_dims = in_dims
        self.out_caps = out_caps
        self.out_dims = out_dims
        self.n_iters = n_iters
        self.caps_dim = caps_dim
        self.learn_coupling = learn_coupling

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)

            if self.learn_coupling:
                b_initializer = initializers._get_initializer(0)
                self.b = variable.Parameter(b_initializer)
            else:
                self.b = None

            self._initialize_params()

    def _initialize_params(self):
        """
        """
        W_shape = (1, self.in_caps, self.out_caps, self.in_dims, self.out_dims)
        self.W.initialize(W_shape)

        if self.b is not None:
            b_shape = (1, self.in_caps, self.out_caps, 1, 1)
            self.b.initialize(b_shape)

    def __call__(self, x):
        """ call
        Args:
            x: [batch, n_global_capsule, caps_dim, n_local_grid, n_local_grid]
                ex) [?, 32, 8, 6, 6]
                    -> [?, 32, 6, 6, 8]
                    -> [?, 1152, 10, 8, 1]
        """
        # calculating x_hat
        x = F.swapaxes(x, self.caps_dim, -1)
        x = F.reshape(x, (-1, self.in_caps, self.in_dims))
        x = F.expand_dims(x, -1)
        x = F.expand_dims(x, 2)
        x = F.tile(x, (1, 1, self.out_caps, 1, 1))
        Ws = F.tile(self.W, (x.shape[0], 1, 1, 1, 1))
        x_hats = F.matmul(Ws, x, transa=True)

        # dynamic routing
        bs = self._routing(x_hats.data)
        c_ij = F.softmax(bs, axis=2)
        s_j = F.broadcast_to(c_ij, x_hats.shape) * x_hats
        s_j = F.sum(s_j, axis=1, keepdims=True)
        v_j = squash(s_j, axis=-2)
 
        return F.reshape(v_j, (-1, self.out_caps, self.out_dims))

    def _routing(self, x_hats):
        """ routing
        Args:
            x_hats: [batch, input_n_capsules, output_n_capusles, 1, outut_n_dims]
                ex) [?, 1152, 10, 1, 16]
        Return:
            
        """
        if self.b is not None:
            bs = F.tile(self.b, (x_hats.shape[0], 1 ,1 ,1, 1))
        else:
            bs = self.xp.zeros(
                (x_hats.shape[0], self.in_caps, self.out_caps, 1, 1),
                dtype="float32"
            )

        for _ in range(self.n_iters):
            c_ij = F.softmax(bs, axis=2)
            s_j = F.broadcast_to(c_ij, x_hats.shape) * x_hats
            s_j = F.sum(s_j, axis=1, keepdims=True)
            v_j = squash(s_j, axis=-2)
            bs = bs + F.matmul(x_hats, F.broadcast_to(v_j, x_hats.shape), transa=True)
        
        return bs
