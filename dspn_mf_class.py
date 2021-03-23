# Code adapted from:
# Salakhutdinov, R. and Hinton, G. Deep Boltzmann Machines. AISTATS, 2009.
import sys
import numpy as np

def mf_class(data,vishid,hidbiases,visbiases,hidpen,penbiases):

    [numdim, numhid]=vishid.shape
    [numhid, numpen]=hidpen.shape

    numcases = data.shape[0]
    bias_hid = np.tile(hidbiases, (numcases, 1))
    bias_pen = np.tile(penbiases, (numcases, 1))
    big_bias = np.dot(data, vishid)

    temp_h1 = 1.0/(1 + np.exp(np.dot(-1*data, (2*vishid)) -
                              np.tile(hidbiases, (numcases,1))))
    temp_h2 = 1.0/(1 + np.exp(np.dot(-temp_h1, hidpen) - bias_pen))

    for ii in range(50):
        totin_h1 = big_bias + bias_hid + np.dot(temp_h2,hidpen.T)
        temp_h1_new = 1.0/(1 + np.exp(-totin_h1))

        totin_h2 =  np.dot(temp_h1_new, hidpen) + bias_pen
        temp_h2_new = 1.0/(1 + np.exp(-totin_h2))

        diff_h1 = np.sum(np.abs(temp_h1_new - temp_h1))/(numcases*numhid)
        diff_h2 = np.sum(np.abs(temp_h2_new - temp_h2))/(numcases*numpen)
        #   fprintf(1,'\t\t\t\tii=%d h1=%f h2=%f\r',ii,diff_h1,diff_h2);
        if (diff_h1 < 0.0000001 and diff_h2 < 0.0000001):
           break

        temp_h1 = temp_h1_new
        temp_h2 = temp_h2_new


    temp_h1 = temp_h1_new
    temp_h2 = temp_h2_new

    return temp_h1, temp_h2

