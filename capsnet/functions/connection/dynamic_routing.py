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
    bs = xp.zeros(shape, dtype='f')
    for i_iter in range(n_iters):
        cs = F.softmax(bs, axis=1)
        Cs = F.broadcast_to(cs[:, :, None], x_hats.shape)

        if i_iter == (n_iters - 1):
            ss = F.sum(Cs * x_hats, axis=3)
            vs = squash(ss, axis=2)
        else:
            ss = F.sum(Cs * x_hats.data, axis=3)
            vs = squash(ss, axis=2)
            Vs = F.broadcast_to(vs[:, :, :, None], x_hats.shape)
            bs = bs + F.sum(Vs * x_hats.data, axis=2)

    return vs
