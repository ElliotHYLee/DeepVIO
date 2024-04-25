from torch.utils.data import Dataset, DataLoader
from src.DataReader.CNN_Data.PrepData import DataManager
import numpy as np
import cv2
import time
from sklearn.utils import shuffle

class VODataSetManager_CNN():
    def __init__(self, dsName='airsim', subType='mr', seq=[0], isTrain=True, split=0.2):
        data = DataManager()
        data.initHelper(dsName, subType, seq)
        data.standardizeImgs(isTrain)
        data.standardizeGyro(isTrain)

        if dsName == 'agz' and isTrain:
            idx = np.arange(0, 2000, 1)
        else:
            idx = np.arange(0, data.numTotalData, 1)
        N = idx.shape[0]
        if isTrain:
            idx = shuffle(idx)
            valN = int(N * split)
            trainN = N - valN
            trainIdx = idx[0:trainN]
            valIdx = idx[trainN:]
            self.trainSet = VODataSet_CNN(trainN, trainIdx)
            self.valSet = VODataSet_CNN(valN, valIdx)
        else:
            self.testSet = VODataSet_CNN(N, idx)

class VODataSet_CNN(Dataset):
    def __init__(self, N, idxList):
        self.dm = DataManager()
        self.N = N
        self.idxList = idxList

    def __getitem__(self, i):
        index = self.idxList[i]
        try:
            return self.dm.imgs[index], self.dm.imgs[index+1], \
                   self.dm.du[index], \
                   self.dm.dw[index], \
                   self.dm.dw_gyro[index], \
                   self.dm.gyro_standard[index], \
                   self.dm.dtrans[index],\
                   self.dm.dtr_gnd[index],\
                   self.dm.rotM_bdy2gnd[index]
        except:
            print('this is an error @ VODataSet_CNN of VODataSet.py')
            print(self.dm.imgs.shape)
            print(i, index)

    def __len__(self):
        return self.N


if __name__ == '__main__':
    start = time.time()
    dm = VODataSetManager_CNN(dsName='kitti', subType='none', seq=[0, 2, 4, 6], isTrain=False)
    print(time.time() - start)
    #trainSet, valSet = dm.trainSet, dm.valSet
    dataSet = dm.testSet
    trainLoader = DataLoader(dataset = dataSet, batch_size=64)
    sum = 0
    for batch_idx, (img0, img1, du, dw, dw_gyro, gyro_standard, dtrans, dtr_gnd, rotM_bdy2gnd) in enumerate(trainLoader):
        img0 = img0.data.numpy()
        img1 = img1.data.numpy()
        sum += img0.shape[0]

        for i in range (img0.shape[0]):
            img_t0 = img0[i,:]
            img_t1 = img1[i,:]
            img_t0 = np.reshape(img_t0, (360, 720, 3))
            img_t1 = np.reshape(img_t1, (360, 720, 3))
            imgcon = img_t1 - img_t0
            cv2.imshow('img0', img_t0)
            cv2.imshow('img1', img_t1)
            cv2.imshow('imgcon', imgcon)
            key = cv2.waitKey(1) & 0xFF  # Use mask for compatibility with 64-bit machines
            if key == ord('q'):  # Check if 'q' was pressed
                break