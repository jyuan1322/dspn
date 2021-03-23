import numpy as np

def deconvert(X, numdims, numhids, numpens, numlab):
    # NOTE: see 'reshape'
    # https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    X = np.squeeze(X)
    w1_vishid = X[:numdims * numhids].reshape((numdims, numhids), order='F').copy()
    xxx = numdims * numhids
    w1_penhid = X[xxx:xxx + numpens * numhids].reshape((numpens, numhids), order='F').copy()
    xxx = xxx + numpens * numhids
    hidpen = X[xxx:xxx + numhids * numpens].reshape((numhids, numpens), order='F').copy()
    xxx = xxx + numhids * numpens
    w_class = X[xxx:xxx + numpens * numlab].reshape((numpens, numlab), order='F').copy()
    xxx = xxx + numpens * numlab
    hidbiases = X[xxx:xxx + numhids].reshape((1, numhids), order='F').copy()
    xxx = xxx + numhids
    penbiases = X[xxx:xxx + numpens].reshape((1, numpens), order='F').copy()
    xxx = xxx + numpens
    topbiases = X[xxx:xxx + numlab].reshape((1, numlab), order='F').copy()
    xxx = xxx + numlab
    return w1_vishid, w1_penhid, hidpen, w_class, hidbiases, penbiases, topbiases

def cg(VV, Dim, XX, target, temp_h2, cg_init=False):
    numdims, numhids, numpens, numlab = Dim
    N = XX.shape[0]

    X=VV
    # Do deconversion.
    w1_vishid, w1_penhid, hidpen, w_class, hidbiases, penbiases, topbiases = \
        deconvert(X, numdims, numhids, numpens, numlab)

    bias_hid = np.tile(hidbiases, (N,1))
    bias_pen = np.tile(penbiases, (N,1))
    bias_top = np.tile(topbiases, (N,1))

    w1probs = 1.0/(1 + np.exp(np.dot(-1*XX, w1_vishid) - np.dot(temp_h2, w1_penhid) - bias_hid))
    w2probs = 1.0/(1 + np.exp(np.dot(-1*w1probs, hidpen) - bias_pen))
    targetout = np.exp(np.dot(w2probs, w_class) + bias_top)
    targetout = np.divide(targetout, np.tile(np.sum(targetout, axis=1)[:,np.newaxis], (1, numlab)))

    f = -np.sum(np.multiply(target, np.log(targetout)))

    IO = targetout - target
    Ix_class=IO
    dw_class = np.dot(w2probs.T, Ix_class)
    dtopbiases = np.sum(Ix_class, axis=0)

    Ix2 = np.multiply(np.multiply(np.dot(Ix_class,
                                         w_class.T),
                                  w2probs),
                      (1-w2probs))
    dw2_hidpen = np.dot(w1probs.T, Ix2)
    dw2_biases = np.sum(Ix2, axis=0)

    Ix1 = np.multiply(np.multiply(np.dot(Ix2,
                                         hidpen.T),
                                  w1probs),
                      (1-w1probs))
    dw1_penhid = np.dot(temp_h2.T, Ix1)

    dw1_vishid = np.dot(XX.T, Ix1)
    dw1_biases = np.sum(Ix1, axis=0)

    if cg_init:
        # dhidpen = 0*dw2_hidpen # not used?
        dw1_penhid = 0*dw1_penhid
        dw1_vishid = 0*dw1_vishid
        dw2_biases = 0*dw2_biases
        dw1_biases = 0*dw1_biases

    df = np.concatenate((dw1_vishid.flatten(order='F'),
                         dw1_penhid.flatten(order='F'),
                         dw2_hidpen.flatten(order='F'),
                         dw_class.flatten(order='F'),
                         dw1_biases.flatten(order='F'),
                         dw2_biases.flatten(order='F'),
                         dtopbiases.flatten(order='F')), axis=None)
    return f, df





