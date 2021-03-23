import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from dspn_backprop import dspn_backprop

if __name__=="__main__":
    data_dir = r"C:/Users/Jie Yuan/Downloads/INT-20_DSPNcode_SCZ/DSPN_scz_large"

    tag = 'scz'
    qt = 1
    numhid = 400   # number of hidden units
    numpen = 100

    nPerBatch = 640 # prop
    nBatch = 1
    maxepoch = 50  # maximum number of epochs

    errsAll = []

    XY = set({})
    with open(data_dir + os.path.sep + 'datasets' +
              os.path.sep + 'XYgenes.txt') as f:
        for line in f:
           XY.add(line.strip())


    for modelIt in range(1,11):

        data_mat = loadmat(data_dir + os.path.sep +
                           'datasets' + os.path.sep +
                           'scz_data%s.mat' % (modelIt))
        init_mat = loadmat(data_dir + os.path.sep +
                           'init' + os.path.sep +
                           'model%s_init.mat' % (modelIt))
        # print(data_mat.keys())

        numlab = 2
        batchdata = []
        testdata = []
        nMF = 20 # for imputt

        # if ~exist('nData_te')
        #     nData_te = nData;
        # end
        nData = np.squeeze(data_mat['nData'])
        nData_te = np.squeeze(data_mat['nData_te'])
        X_Trait_tr = np.array(data_mat['X_Trait_tr'])
        X_Trait_te = np.array(data_mat['X_Trait_te'])

        X_SNP_tr = np.zeros((nData, 2))
        X_SNP_te = np.zeros((nData_te, 2))
        nSNP = 2

        X_geneIds2 = data_mat['X_geneIds2']
        X_geneIds2 = [x[0] for x in X_geneIds2.flatten()]
        filt = [int(x not in XY) for x in X_geneIds2]
        X_Gene_tr = data_mat['X_Gene_tr'][:,np.where(filt)[0]]
        X_Gene_te = data_mat['X_Gene_te'][:,np.where(filt)[0]]
        sc = data_mat['sc'].flatten()[np.where(filt)[0]]

        batchdata = np.empty((nPerBatch,X_Gene_tr.shape[1],nBatch))
        batchdata_snp = np.empty((nPerBatch,X_SNP_tr.shape[1],nBatch))
        batchdata_trait = np.empty((nPerBatch,X_Trait_tr.shape[1],nBatch))
        traindata = X_Gene_tr[:nPerBatch * nBatch,:]
        traindata_snp = X_SNP_tr[:nPerBatch * nBatch,:]
        traindata_trait = X_Trait_tr[:nPerBatch * nBatch,:]
        for i in range(nBatch):
            batchdata[:,:,i] = X_Gene_tr[:nPerBatch,:]
            batchdata_snp[:,:,i] = X_SNP_tr[:nPerBatch,:]
            batchdata_trait[:,:,i] = X_Trait_tr[:nPerBatch,:]
            X_Gene_tr = X_Gene_tr[nPerBatch:,:]
            X_Trait_tr = X_Trait_tr[nPerBatch:,:]
            X_SNP_tr = X_SNP_tr[nPerBatch:,:]

        testdata = X_Gene_te
        testdata_snp = X_SNP_te
        testdata_trait = X_Trait_te

        X_train0 = traindata
        y_train = traindata_trait[:, 1]
        nGene = X_train0.shape[1]
        cut = int(np.ceil(nGene * (qt / 100)))
        sorted_sc = sorted(enumerate(sc), key=lambda x: x[1], reverse=True)
        dum = np.array([x[1] for x in sorted_sc])
        pmt = np.array([x[0] for x in sorted_sc])
        traindata = traindata[:, pmt]
        batchdata = batchdata[:, pmt,:]
        testdata = testdata[:, pmt]
        traindata = traindata[:, :cut]
        batchdata = batchdata[:, :cut,:]
        testdata = testdata[:, :cut]

        vishid = init_mat['vishid']
        hidpen = init_mat['hidpen']
        hidbiases = init_mat['hidbiases']
        visbiases = init_mat['visbiases']
        penbiases = init_mat['penbiases']

        test_err, train_err = dspn_backprop(batchdata, batchdata_trait, testdata, testdata_trait,
                                            vishid, hidpen, hidbiases, visbiases, penbiases,
                                            maxepoch, data_dir)
        errsAll.append(min(test_err))
        plt.plot(range(1, len(train_err) + 1), train_err, label='train set')
        plt.plot(range(1, len(test_err)+1), test_err, label='test set')
        plt.xlabel('iteration')
        plt.ylabel('label accuracy')
        plt.legend()
        plt.show()