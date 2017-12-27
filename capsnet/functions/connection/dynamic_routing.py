import chainer.functions as F
from chainer import cuda
from functions.activation.squash import squash


def routing(x_hats, n_iters):
    """
    Args:
        x_hats:
            [batch, out_caps, out_caps_dim, in_caps, *]
    """
    xp = cuda.get_array_module(x_hats)
    n_batch = x_hats.shape[0]
    out_caps = x_hats.shape[1]
    in_caps = x_hats.shape[3]

    if len(x_hats.shape) > 3:
        extra_dim = x_hats.shape[4:]
        shape = (n_batch, out_caps, in_caps, *extra_dim)
    else:
        shape = (n_batch, out_caps, in_caps)

    b_ij = xp.zeros(shape, dtype='f')
    for i_iter in range(n_iters):
        c_ij = F.softmax(b_ij, axis=1)
        c_tmp = F.broadcast_to(c_ij[:, :, None], x_hats.shape)

        if i_iter == (n_iters - 1):
            s_j = F.sum(c_tmp * x_hats, axis=3)
            v_j = squash(s_j, axis=2)
        else:
            s_j = F.sum(c_tmp * x_hats.data, axis=3)
            v_j = squash(s_j, axis=2)
            Vs = F.broadcast_to(v_j[:, :, :, None], x_hats.shape)
            b_ij = b_ij + F.sum(Vs * x_hats.data, axis=2)

    return v_j
