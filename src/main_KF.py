from src.DataReader.KF_Data.KF_PrepData import DataManager
from scipy import signal
from src.Params import *
from src.Models.KF_Model.KF_BLock import *
from src.Models.KF_Model.KF_Model import *
import torch.optim as optim
import matplotlib.pyplot as plt
from src.Params import getNoiseLevel

dsName, subType, seq = 'kitti', 'none', [0, 2, 7, 10]

wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType + '_KF'

def preClamp(data):
    if dsName=='kitti':
        return data
    N = data.shape[0]
    for i in range(0, N):
        row = data[i, :]
        for j in range(0, 3):
            val = row[j]
            if val > 1:
                val = 1
            elif val < -1:
                val = -1
            row[j] = val
        data[i] = row
    return data

def filtfilt(data):
    y = np.zeros_like(data)
    b, a = signal.butter(8, 0.1)
    for i in range(0, 3):
        y[:, i] = signal.filtfilt(b, a, data[:, i], padlen=100)
    return y

def plotter(filt, gt, mSignal):

    plt.figure()
    plt.subplot(311)
    plt.title("Vel GT vs Vel Filt")
    plt.plot(gt[:, 0], 'r.')
    plt.plot(mSignal[:, 0], 'b.', markersize=0.5)   
    plt.plot(filt[:, 0], 'g.', markersize=0.8)
    plt.subplot(312)
    plt.plot(gt[:, 1], 'r')
    plt.plot(mSignal[:, 1], 'b.', markersize=0.5)
    plt.plot(filt[:, 1], 'g.', markersize=0.8)
    plt.subplot(313)
    plt.plot(gt[:, 2], 'r')
    plt.plot(mSignal[:, 2], 'b.', markersize=0.5)
    plt.plot(filt[:, 2], 'g.', markersize=0.8)

    posFilt = integrate(filt)
    posGT = integrate(gt)
    posMeas = integrate(mSignal)
    plt.figure()
    plt.suptitle("Pose GT vs Pose Filt XYZ")
    plt.subplot(311)
    plt.plot(posGT[:, 0], 'r')
    plt.plot(posMeas[:, 0], 'b')
    plt.plot(posFilt[:, 0], 'g')
    plt.subplot(312)
    plt.plot(posGT[:, 1], 'r')
    plt.plot(posMeas[:, 1], 'b')
    plt.plot(posFilt[:, 1], 'g')
    plt.subplot(313)
    plt.plot(posGT[:, 2], 'r')
    plt.plot(posMeas[:, 2], 'b')
    plt.plot(posFilt[:, 2], 'g')

    plt.figure()
    plt.title("Pose GT vs Pose Filt 2D")
    plt.plot(posGT[:, 0], posGT[:, 2], 'r')
    plt.plot(posMeas[:, 0], posMeas[:, 2], 'b')
    plt.plot(posFilt[:, 0], posFilt[:, 2], 'g')

    return posFilt, posGT

def prepData(seqLocal = seq):
    dm = DataManager()
    dm.initHelper(dsName, subType, seqLocal)
    dt = dm.dt

    acc_gnd = dm.acc_gnd #dm.accdt_gnd
    acc_gnd = preClamp(acc_gnd)

    dtr_gnd = dm.pr_dtr_gnd
    dtr_gnd = preClamp((dtr_gnd))

    dtr_covm_gnd = dm.dtr_cov_gnd

    gt_dtr_gnd = preClamp(dm.gt_dtr_gnd)
    gt_dtr_gnd = filtfilt(gt_dtr_gnd)
    return gt_dtr_gnd, dt, acc_gnd, dtr_gnd, dtr_covm_gnd

def main(isTrain=False):
    kfNumpy = KFBlock()
    gt_dtr_gnd, dt, acc_gnd, dtr_gnd, dtr_covm_gnd = prepData(seqLocal=seq)
    posGT = np.cumsum(gt_dtr_gnd, axis=0)
    gnet = GuessNet()

    if not isTrain:
        #gnet.train()
        checkPoint = torch.load(wName + '.pt')
        gnet.load_state_dict(checkPoint['model_state_dict'])
        gnet.load_state_dict(checkPoint['optimizer_state_dict'])
    else:
        gnet.eval()

    kf = TorchKFBLock(gt_dtr_gnd, dt, acc_gnd, dtr_gnd, dtr_covm_gnd)
    rmser = GetRMSE()
    optimizer = optim.RMSprop(gnet.parameters(), lr=10 ** -4)

    if isTrain:
        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()

    iterN = 50 if isTrain else 1
    for epoch in range(0, iterN):
        guess, sign = gnet()
        filt = kf.forward(guess, sign)
        velRMSE, posRMSE = rmser(filt, gt_dtr_gnd)
        params = guess.data.numpy()
        paramsSign = sign.data.numpy()
        loss = posRMSE.data.numpy() + velRMSE.data.numpy()
        theLOss = velRMSE + posRMSE
        if isTrain:
            if epoch == 10:
                optimizer = optim.RMSprop(gnet.parameters(), lr=10 ** -4)
            optimizer.zero_grad()
            theLOss.backward(torch.ones_like(posRMSE))
            optimizer.step()

            temp = filt.data.numpy()
            posKF = np.cumsum(temp, axis=0)

            fig.clear()
            plt.subplot(311)
            plt.plot(posGT[:, 0], 'r')
            plt.plot(posKF[:, 0], 'b')
            plt.subplot(312)
            plt.plot(posGT[:, 1], 'r')
            plt.plot(posKF[:, 1], 'b')
            plt.subplot(313)
            plt.plot(posGT[:, 2], 'r')
            plt.plot(posKF[:, 2], 'b')
            plt.pause(0.001)
            fig.canvas.draw()
            plt.savefig('KFOptimHistory/'+dsName +' ' + subType + ' temp ' + str(epoch) + '.png')

        #if np.mod(epoch, 10):
        print('epoch: %d' % epoch)
        print('params: ')
        print(params)
        print(paramsSign)
        print('posRMSE: %.4f, %.4f, %.4f' %(loss[0], loss[1], loss[2]))

    torch.save({
        'model_state_dict': gnet.state_dict(),
        'optimizer_state_dict': gnet.state_dict(),
    }, wName + '.pt')

    if isTrain:
        kfRes = filt.data.numpy()
        # _, _ = plotter(kfRes, gt_dtr_gnd)
    else:
        noise = getNoiseLevel()
        for ii in range(5, 6):
            gt_dtr_gnd, dt, acc_gnd, dtr_gnd, dtr_covm_gnd = prepData(seqLocal=[ii])
            kfNumpy.setR(params, paramsSign)
            kfRes = kfNumpy.runKF(dt, acc_gnd, dtr_gnd, dtr_covm_gnd)
            posFilt, posGT = plotter(kfRes, gt_dtr_gnd, dtr_gnd)
            np.savetxt('Results/Data/posFilt' + str(ii) + '_' + str(noise) + '.txt', posFilt)
            np.savetxt('Results/Data/posGT' + str(ii) + '_' + str(noise) +  '.txt', posGT)

    plt.show()


if __name__ == '__main__':
    #main(isTrain=True)
    main(isTrain=False)


