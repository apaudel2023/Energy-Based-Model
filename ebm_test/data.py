import numpy as np
from keras import backend as K

def potential_fn(dataset):
    """
    toy potention functions
    Code borrowed from https://github.com/kamenbliznashki/normalizing_flows/blob/master/bnaf.py"""
    w1 = lambda z: K.sin(2 * np.pi * z[:, 0] / 4)
    w2 = lambda z: 3 * K.exp(-0.5 * ((z[:, 0] - 1) / 0.6)**2)
    w3 = lambda z: 3 * K.sigmoid((z[:, 0] - 1) / 0.3)

    if dataset == 'u1':
        return lambda z: 0.5 * ((K.sqrt(K.sum(z**2, axis=1)) - 2) / 0.4)**2 - \
                           K.log(K.exp(-0.5*((z[:, 0] - 2) / 0.6)**2) + K.exp(-0.5*((z[:, 0] + 2) / 0.6)**2) + 1e-10)

    elif dataset == 'u2':
        return lambda z: 0.5 * ((z[:, 1] - w1(z)) / 0.4)**2

    elif dataset == 'u3':
        return lambda z: - K.log(K.exp(-0.5*((z[:, 1] - w1(z))/0.35)**2) + \
                                K.exp(-0.5*((z[:, 1] - w1(z) + w2(z))/0.35)**2) + 1e-10)

    elif dataset == 'u4':
        return lambda z: - K.log(K.exp(-0.5*((z[:, 1] - w1(z))/0.4)**2) + \
                                K.exp(-0.5*((z[:, 1] - w1(z) + w3(z))/0.35)**2) + 1e-10)

    else:
        raise RuntimeError('Invalid potential name to sample from.')

def sample_2d_data(dataset, n_samples):
    """generate samples from 2D toy distributions
    Code borrowed from https://github.com/kamenbliznashki/normalizing_flows/blob/master/bnaf.py"""
    z = np.random.randn(n_samples, 2)

    if dataset == '8gaussians':
        scale = 4
        sq2 = 1/np.sqrt(2)
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (sq2, sq2), (-sq2, sq2), (sq2, -sq2), (-sq2, -sq2)]
        centers = np.array([(scale * x, scale * y) for x, y in centers])
        return sq2 * (0.5 * z + centers[np.random.randint(len(centers), size=(n_samples,))])

    elif dataset == '2spirals':
        n = np.sqrt(np.random.rand(n_samples // 2)) * 540 * (2 * np.pi) / 360
        d1x = - np.cos(n) * n + np.random.rand(n_samples // 2) * 0.5
        d1y = np.sin(n) * n + np.random.rand(n_samples // 2) * 0.5
        x = np.concatenate([np.stack([d1x, d1y], axis=1),
                            np.stack([-d1x, -d1y], axis=1)], axis=0) / 3
        return x + 0.1 * z

    elif dataset == 'checkerboard':
        x1 = np.random.rand(n_samples) * 4 - 2
        x2_ = np.random.rand(n_samples) - np.random.randint(0, 2, (n_samples,)) * 2
        x2 = x2_ + np.floor(x1) % 2
        return np.stack([x1, x2], axis=1) * 2

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        linspace4 = np.linspace(0, 2 * np.pi, n_samples4 + 1)[:-1]
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3 + 1)[:-1]
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2 + 1)[:-1]
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1 + 1)[:-1]

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        x = np.stack([np.concatenate([circ4_x, circ3_x, circ2_x, circ1_x]),
                      np.concatenate([circ4_y, circ3_y, circ2_y, circ1_y])], axis=1) * 3.0

        x = x[np.random.randint(0, n_samples, size=(n_samples,))]

        return x + np.random.normal(loc=0.0, scale=0.08, size=x.shape)

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')
