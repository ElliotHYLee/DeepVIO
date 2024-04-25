from src.DataReader.CNN_Data.ReadData_CNN import *
import time
from src.Params import branchName
##############################################################################################
## Rule of thumb: don't call any other function to reduce lines of code with the img data in np.
## Otherwise, it could cause memeory dupilication.
##############################################################################################
class Singleton:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

class DataManager(Singleton):

    def initHelper(self, dsName='kitti', subType='mr', seq=[1, 3, 5]):
        self.dsName = dsName
        self.numChannel = 3 if self.dsName is not 'euroc' else 1
        self.subType = subType
        self.numDataset = len(seq)
        dataObj = [ReadData_CNN(dsName, subType, seq[i]) for i in range(0, self.numDataset)]

        # get number of data points
        self.numDataList = [dataObj[i].numData for i in range(0, self.numDataset)]
        self.numTotalData = np.sum(self.numDataList)
        self.numTotalImgData = np.sum([dataObj[i].numImgs for i in range(0, self.numDataset)])
        print(self.numDataList)
        print(self.numTotalData)

        # numeric data
        print('numeric data concat')
        self.dt = np.concatenate([dataObj[i].dt for i in range(0, self.numDataset)], axis=0)
        self.du = np.concatenate([dataObj[i].du for i in range(0, self.numDataset)], axis=0)
        self.dw = np.concatenate([dataObj[i].dw for i in range(0, self.numDataset)], axis=0)
        self.dw_gyro = np.concatenate([dataObj[i].dw_gyro for i in range(0, self.numDataset)], axis=0)
        self.dtrans = np.concatenate([dataObj[i].dtr for i in range(0, self.numDataset)], axis=0)
        self.dtr_gnd = np.concatenate([dataObj[i].dtr_gnd for i in range(0, self.numDataset)], axis=0)
        self.pos_gnd = np.concatenate([dataObj[i].pos_gnd for i in range(0, self.numDataset)], axis=0)
        self.rotM_bdy2gnd = np.concatenate([dataObj[i].rotM_bdy2gnd for i in range(0, self.numDataset)], axis=0)
        self.acc_gnd = np.concatenate([dataObj[i].acc_gnd for i in range(0, self.numDataset)], axis=0)
        print('done numeric data concat')

        # img data
        print('img data concat')
        self.numTotalImgs = sum([dataObj[i].numImgs for i in range(0, self.numDataset)])
        self.imgs = np.zeros((self.numTotalImgData, self.numChannel, 360, 720), dtype=np.float32)
        s, f = 0, 0
        for i in range(0, self.numDataset):
            temp = dataObj[i].numImgs
            f = s + temp
            self.imgs[s:f, :] = dataObj[i].imgs
            dataObj[i] = None
            s = f
        dataObj = None
        print('done img data concat')

    def standardizeGyro(self, isTrain):
        print('standardizing gyro')
        normPath = 'Norms/' + branchName() + '_' + self.dsName + '_' + self.subType
        if isTrain:
            gyroMean = np.mean(self.dw_gyro, axis=0)
            gyroStd = np.std(self.dw_gyro, axis=0)
            np.savetxt(normPath + 'gyroMean.txt', gyroMean)
            np.savetxt(normPath + 'gyroStd.txt', gyroStd)
        else:
            gyroMean = np.loadtxt(normPath + 'gyroMean.txt')
            gyroStd = np.loadtxt(normPath + 'gyroStd.txt')
        self.gyro_standard = self.dw_gyro - gyroMean
        self.gyro_standard = np.divide(self.gyro_standard, gyroStd).astype(np.float32)

    def standardizeImgs(self, isTrain):
        print('preparing to standardize imgs')
        mean = np.mean(self.imgs, axis=(0, 2, 3))
        std = np.std(self.imgs, axis=(0, 2, 3))
        normPath = 'Norms/' + branchName() + '_' + self.dsName + '_' + self.subType
        if isTrain:
            np.savetxt(normPath + '_img_mean.txt', mean)
            np.savetxt(normPath + '_img_std.txt', std)
        else:
            mean = np.loadtxt(normPath + '_img_mean.txt')
            std = np.loadtxt(normPath + '_img_std.txt')
            if self.dsName == 'euroc':
                mean = np.reshape(mean, (1,1))
                std = np.reshape(std, (1,1))

        # standardize imgs
        print('standardizing imgs')
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        for i in range(0, self.imgs.shape[1]):
            self.imgs[:, i, :, :] = (self.imgs[:, i, :, :] - mean[i])/std[i]
        print('done standardizing imgs')

if __name__ == '__main__':
    s = time.time()
    m = DataManager()
    m.initHelper(dsName='kitti', subType='mr', seq=[0])
    
    for i in range(0, m.numTotalImgData):
        img = m.imgs[i, :]
        img = np.reshape(img, (360, 720, m.numChannel))
        cv2.imshow('check', img)
        # print(img)
        key = cv2.waitKey(1) & 0xFF  # Use mask for compatibility with 64-bit machines
        if key == ord('q'):  # Check if 'q' was pressed
            break
  