import numpy as np
import minimize
from scipy.io import loadmat
from cg_functions import cg
from dspn_mf_class import mf_class

"""
def testfun(x):
    f = x*x
    df = 2*x
    return f,df

max_iter=5
VV=5
[X, fX, num_iter] = minimize.run(testfun, VV, args=(), length=max_iter)
print(X)
print(fX)
print(num_iter)
"""
"""
import scipy.io as sio
bdict = sio.loadmat('test.mat')
VV = bdict['VV'].squeeze()
print(VV.shape)
Dim = bdict['Dim'].squeeze()
Dim = np.array(Dim, dtype=np.int)
print(Dim)
data = bdict['data']
targets = bdict['targets']
temp_h2 = bdict['temp_h2']
f, df = cg(VV, Dim, data, targets, temp_h2, cg_init=False)
print(f)
print(df[15544:15552])
print(df.shape)
"""


data_mat = loadmat('test.mat')
VV = data_mat['VV'].squeeze()
Dim = np.squeeze(data_mat['Dim']).tolist()
data = data_mat['data']
targets = data_mat['targets']
temp_h2 = data_mat['temp_h2']
max_iter = np.squeeze(data_mat['max_iter'])
[X, fX, num_iter] = minimize.run(cg, VV, args=(Dim, data, targets, temp_h2, False), length=max_iter)
print(np.sum(X-VV))
print(fX)


"""
data_mat = loadmat('test2.m')
data = data_mat['data']
vishid = data_mat['vishid']
hidbiases = data_mat['hidbiases']
visbiases = data_mat['visbiases']
hidpen = data_mat['hidpen']
penbiases = data_mat['penbiases']
temp_h1, temp_h2 = mf_class(data,vishid,hidbiases,visbiases,hidpen,penbiases)
print(temp_h1[0,:10])
print(temp_h2[0,:10])
"""