# Code adapted from:
# Salakhutdinov, R. and Hinton, G. Deep Boltzmann Machines. AISTATS, 2009.
import sys, os, pickle
import numpy as np
import minimize
from dspn_mf_class import mf_class
from cg_functions import cg, deconvert

# data_dir = r"C:/Users/Jie Yuan/Downloads/INT-20_DSPNcode_SCZ/DSPN_scz_large"

def calc_error_preds(ttdata, ttdata_trait, temp_h2_tt,
                     w1_vishid, w1_penhid, w2, w_class,
                     h1_biases, h2_biases, topbiases, numlab):
    # tt for test/train
    [ttnumcases, ttnumdims, ttnumbatches] = ttdata.shape
    N = ttnumcases
    bias_hid = np.tile(h1_biases, (N, 1))
    bias_pen = np.tile(h2_biases, (N, 1))
    bias_top = np.tile(topbiases, (N, 1))

    err = 0
    err_cr = 0
    counter = 0
    totCount = 0
    predVec = []
    y_tt = []
    for batch in range(ttnumbatches):
        data = ttdata[:, :, batch]
        temp_h2 = temp_h2_tt[:, :, batch]
        target = ttdata_trait[:, :, batch]

        w1probs = 1.0 / (1 + np.exp(np.dot(-1 * data, w1_vishid) -
                                    np.dot(temp_h2, w1_penhid) - bias_hid))
        w2probs = 1.0 / (1 + np.exp(np.dot(-1 * w1probs, w2) - bias_pen))
        targetout = np.exp(np.dot(w2probs, w_class) + bias_top)
        targetout = np.divide(targetout, np.tile(np.sum(targetout, axis=1)[:, np.newaxis], (1, numlab)))
        J = np.argmax(targetout, axis=1)
        J1 = np.argmax(target, axis=1)
        predVec = np.concatenate((predVec, J), axis=None)  # None: arrays flattened
        y_tt = np.concatenate((y_tt, J1), axis=None)
        counter += np.sum((J != J1).astype(int))
        err_cr -= np.sum(np.multiply(target, np.log(targetout)))
        totCount += len(J)

    err_rate = counter / totCount
    return predVec, y_tt, err_rate, err_cr

def dspn_backprop(batchdata, batchdata_trait, testdata, testdata_trait,
                  vishid, hidpen, hidbiases, visbiases, penbiases,
                  maxepoch, data_dir):
    test_err = []
    test_crerr = []
    train_err = []
    train_crerr = []

    [numcases, numdims, numbatches] = batchdata.shape
    N = numcases

    [numdims, numhids] = vishid.shape
    [numhids, numpens] = hidpen.shape

    ### Preprocess the data ###
    if len(testdata_trait.shape)==2:
        testdata_trait = testdata_trait[:,:,np.newaxis]
    if len(testdata.shape)==2:
        testdata = testdata[:,:,np.newaxis]
    [testnumcases, testnumdims, testnumbatches] = testdata.shape
    N = testnumcases
    temp_h2_test = np.zeros((testnumcases, numpens, testnumbatches))
    for batch in range(testnumbatches):
        data = testdata[:,:, batch]
        [temp_h1, temp_h2] = mf_class(data, vishid, hidbiases, visbiases, hidpen, penbiases)
        temp_h2_test[:,:, batch] = temp_h2

    if len(batchdata.shape)==2:
        batchdata = batchdata[:,:,np.newaxis]
    [numcases, numdims, numbatches] = batchdata.shape
    N=numcases
    temp_h2_train = np.zeros((numcases, numpens, numbatches))
    for batch in range(numbatches):
        data = batchdata[:, :, batch]
        [temp_h1, temp_h2] = mf_class(data, vishid, hidbiases, visbiases, hidpen, penbiases)
        temp_h2_train[:, :, batch] = temp_h2

    ###### ###### ###### ###### ###### ######

    w1_penhid = hidpen.T
    w1_vishid = vishid
    w2 = hidpen
    h1_biases = hidbiases
    h2_biases = penbiases
    numlab = testdata_trait.shape[1]

    w_class = 0.1 * np.random.normal(size=(numpens, numlab))
    topbiases = 0.1 * np.random.normal(size=numlab)

    preds = []

    w1_vishid_all = np.empty((w1_vishid.shape[0], w1_vishid.shape[1], maxepoch))
    w2_all = np.empty((w2.shape[0], w2.shape[1], maxepoch))
    w_class_all = np.empty((w_class.shape[0], w_class.shape[1], maxepoch))

    for epoch in range(maxepoch):
        #### TEST STATS
        #### Error rates
        predVec, y_test, err_rate, err_cr = calc_error_preds(testdata, testdata_trait, temp_h2_test,
                                                             w1_vishid, w1_penhid, w2, w_class,
                                                             h1_biases, h2_biases, topbiases, numlab)
        preds.append(predVec)
        test_err.append(err_rate)
        test_crerr.append(err_cr)

        ###### ###### ###### ###### ###### ######

        #### TRAINING STATS
        #### Error rates
        _, _, train_err_rate, train_err_cr = calc_error_preds(batchdata, batchdata_trait, temp_h2_train,
                                                              w1_vishid, w1_penhid, w2, w_class,
                                                              h1_biases, h2_biases, topbiases, numlab)
        train_err.append(train_err_rate)
        train_crerr.append(train_err_cr)

        print('epoch %s: train_err %0.4f, test_err %0.4f; train_crerr %0.4f, test_crerr %0.4f' % \
              (epoch, train_err[-1], test_err[-1], train_crerr[-1], test_crerr[-1]))

        ###### ###### ###### ###### ###### ######

        w1_vishid_all[:,:,epoch] = w1_vishid
        w2_all[:,:,epoch] = w2
        w_class_all[:,:,epoch] = w_class

        outFile = data_dir + os.path.sep + 'out' + os.path.sep + 'model%s_backprop.pickle'

        # save(outFile,'w1_vishid','w1_penhid','w2','w_class','h1_biases','h2_biases',...
        # 'topbiases','test_err','test_crerr','train_err','train_crerr','preds','y_test');

        if epoch==maxepoch-1:
            # save(outFile,'w1_vishid_all','w2_all','w_class_all','-append');
            pickle.dump({'w1_vishid':w1_vishid,
                         'w1_penhid':w1_penhid,
                         'w2':w2,
                         'w_class':w_class,
                         'h1_biases':h1_biases,
                         'h2_biases':h2_biases,
                         'topbiases':topbiases,
                         'test_err':test_err,
                         'test_crerr':test_crerr,
                         'train_err':train_err,
                         'train_crerr':train_crerr,
                         'preds':preds,
                         'y_test':y_test,
                         'w1_vishid_all':w1_vishid_all,
                         'w2_all':w2_all,
                         'w_class_all':w_class_all},open(outFile, 'wb'))
            break

        ### Do Conjugate Gradient Optimization

        for batch in range(numbatches):
            # %     fprintf(1,'epoch %d batch %d\r',epoch,batch);

            data = batchdata[:,:,batch]
            temp_h2 = temp_h2_train[:,:,batch]
            targets = batchdata_trait[:,:,batch]

            ###### DO CG with 3 linesearches
            VV = np.concatenate((w1_vishid.flatten(order='F'),
                                 w1_penhid.flatten(order='F'),
                                 w2.flatten(order='F'),
                                 w_class.flatten(order='F'),
                                 h1_biases.flatten(order='F'),
                                 h2_biases.flatten(order='F'),
                                 topbiases.flatten(order='F')), axis=None)
            Dim = [numdims, numhids, numpens, numlab]


            # checkgrad('CG_MNIST_INIT',VV,10^-5,Dim,data,targets);
            max_iter=3
            if epoch<6:
                # [X, fX, num_iter,ecg_XX] = dspn_minimize(VV,'dspn_CG_INIT',max_iter,Dim,data,targets,temp_h2);
                [X, fX, num_iter] = minimize.run(cg, VV, args=(Dim, data, targets, temp_h2, True), length=max_iter)
            else:
                # [X, fX, num_iter,ecg_XX] = dspn_minimize(VV,'dspn_CG',max_iter,Dim,data,targets,temp_h2);
                [X, fX, num_iter] = minimize.run(cg, VV, args=(Dim, data, targets, temp_h2, False), length=max_iter)
            # NOTE: variable names here are slightly different than in deconvert()
            w1_vishid, w1_penhid, w2, w_class, h1_biases, h2_biases, topbiases = \
                deconvert(X, numdims, numhids, numpens, numlab)
    return test_err, train_err