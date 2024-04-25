from src.DataReader.DataUtils import *
import threading, cv2, sys, time
from src.Params import getNoiseLevel
import numpy as np

class ReadData():
    def __init__(self, dsName='kitti', subType='mr', seq=0):
        self.dsName = dsName
        self.subType = subType

        self.path = getPath(dsName, seq=seq, subType=subType)

        # ground truth data
        self.dt = pd.read_csv(self.path + 'dt.txt', sep=',', header=None).values.astype(np.float32)
        self.du = pd.read_csv(self.path + 'du.txt', sep=',', header=None).values.astype(np.float32)
        self.dw = pd.read_csv(self.path + 'dw.txt', sep=',', header=None).values.astype(np.float32)
        self.dw_gyro = pd.read_csv(self.path + 'dw_gyro.txt', sep=',', header=None).values.astype(np.float32)
        self.dtr = pd.read_csv(self.path + 'dtrans.txt', sep=',', header=None).values.astype(np.float32)
        self.dtr_gnd = pd.read_csv(self.path + 'dtrans_gnd.txt', sep=',', header=None).values.astype(np.float32)
        noise = getNoiseLevel()
        fRName = 'linR' + str(noise) + '.txt'

        self.linR = pd.read_csv(self.path + fRName, sep=',', header=None).values.astype(np.float32)
        self.rotM_bdy2gnd = np.zeros((self.linR.shape[0], 3, 3), dtype=np.float32)
        for i in range(0, self.linR.shape[0]):
            self.rotM_bdy2gnd[i, :, :] = np.reshape(self.linR[i, :], (3, 3))

        self.numData = self.du.shape[0]
        self.pos_gnd = pd.read_csv(self.path + 'pos.txt', sep=',', header=None).values.astype(np.float32)
        self.acc_gnd = pd.read_csv(self.path + 'acc_gnd.txt', sep=',', header=None).values.astype(np.float32)


class ReadData_CNN(ReadData):
    def __init__(self, dsName='airsim', subType='mr', seq=0):
        super().__init__(dsName, subType, seq)
        if dsName == 'airsim' or dsName == 'myroom' or dsName == 'mycar':
            print(self.path)
            self.data = pd.read_csv(self.path + 'data.txt', sep=' ', header=None)
            self.time_stamp = self.data.iloc[:, 0].values
        else:
            self.time_stamp = None

        # images
        self.imgNames = getImgNames(self.path, dsName, ts = self.time_stamp, subType=subType)
        print(len(self.imgNames))
        print(self.imgNames[0])
        self.numImgs = len(self.imgNames)
        self.numChannel = 3 if self.dsName is not 'euroc' else 1
        self.imgs = np.zeros((self.numImgs, self.numChannel, 360, 720), dtype=np.float32)
        self.getImages()

        # # special case
        # s = None
        # if dsName == 'euroc' and subType == 'none2':
        #     s = 2
        # if dsName == 'euroc' and subType == 'none3':
        #     s = 3

        # if s is not None:
        #     idx = np.arange(0, self.numData - s, s)
        #     last = np.reshape(np.max(idx), (1,))
        #     imgIdx = np.concatenate((idx, last + s))
        #     self.dt = self.dt[idx]
        #     self.du = self.du[idx]
        #     self.dw = self.dw[idx]
        #     self.dw_gyro = self.dw_gyro[idx]
        #     self.dtr = self.dtr[idx]
        #     self.dtr_gnd = self.dtr_gnd[idx]
        #     self.linR = self.linR[idx]
        #     self.rotM_bdy2gnd = self.rotM_bdy2gnd[idx]
        #     self.pos_gnd = self.pos_gnd[idx]
        #     self.acc_gnd = self.acc_gnd[idx]
        #     self.imgs = self.imgs[imgIdx]
        #     self.numData = idx.shape[0]
        #     self.numImgs = self.imgs.shape[0]

    def getImgsFromTo(self, start, N):
        if start>self.numImgs:
            sys.exit('ReadData-getImgsFromTo: this should be the case')

        end, N = getEnd(start, N, self.numImgs)
        #print('PrepData-reading imgs from %d to %d(): reading imgs' %(start, end))
        for i in range(start, end):
            fName = self.imgNames[i]
            if self.dsName == 'euroc':
                img = cv2.imread(fName, 0) / 255.0
            else:
                img = cv2.imread(fName) / 255.0
            if self.dsName is not 'airsim':
                img = cv2.resize(img, (720, 360))
            img = np.reshape(img.astype(np.float32), (-1, self.numChannel, 360, 720))
            self.imgs[i,:] = img #no lock is necessary
        #print('PrepData-reading imgs from %d to %d(): done reading imgs' % (start, end))

    def getImages(self):
        partN = 500
        nThread = int(self.numImgs/partN) + 1
        print('# of thread reading imgs: %d'%(nThread))
        threads = []
        for i in range(0, nThread):
            start = i*partN
            threads.append(threading.Thread(target=self.getImgsFromTo, args=(start, partN)))
            threads[i].start()

        for thread in threads:
            thread.join() # wait until this thread ends ~ bit of loss in time..

if __name__ == '__main__':
    s = time.time()
    d = ReadData_CNN(dsName='kitti', subType='mr', seq=0)
    print(time.time() - s)

    for i in range(0, d.numImgs):
        img = d.imgs[i,:]
        img = np.reshape(img, (360, 720, d.numChannel))
        cv2.imshow('asdf', img)
        cv2.waitKey(1)