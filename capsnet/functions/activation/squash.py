import chainer.functions as F


def squash(vector, axis=-1):
    """
    """
    vec_squared_norm = F.sum(F.square(vector), axis=axis, keepdims=True)
    scalar_factors = F.sqrt(vec_squared_norm) / (1. + vec_squared_norm)
    scalar_factors = F.broadcast_to(scalar_factors, vector.shape)
    vec_squashed = scalar_factors * vector
    return vec_squashed
