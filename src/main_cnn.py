# %%

from src.DataReader.CNN_Data.VODataSet import VODataSetManager_CNN
from src.Models.CNN_Model.Model_CNN_0 import Model_CNN_0
from src.Models.CNN_Model.CNN_ModelContainer import CNN_ModelContainer
import numpy as np
import time
from src.Params import *

def train(dsName, subType, seq):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    dm = VODataSetManager_CNN(dsName=dsName, subType=subType, seq=seq, isTrain=True, split=0.2)
    mc = CNN_ModelContainer(Model_CNN_0(dsName), wName=wName)
    mc.regress(dm, epochs=20, batch_size=64, shuffle=False)

def test(dsName, subType, seqRange):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    resName = 'Results/Data/' + branchName() + '_' + dsName + '_'
    for seq in range(seqRange[0], seqRange[1]):
        commName = resName + subType + str(seq)
        dm = VODataSetManager_CNN(dsName=dsName, subType=subType, seq=[seq], isTrain=False)

        mc = CNN_ModelContainer(Model_CNN_0(dsName), wName=wName)
        mc.load_weights(wName +'_best', train=False)

        pr_du, du_cov, \
        pr_dw, dw_cov, \
        pr_dtr, dtr_cov, \
        pr_dtr_gnd = mc.predict(dm, batch_size=64)

        noise = getNoiseLevel()

        np.savetxt(commName + '_du' + str(noise) + '.txt', pr_du)
        np.savetxt(commName + '_du_cov' + str(noise) + '.txt', du_cov)
        np.savetxt(commName + '_dw' + str(noise) + '.txt', pr_dw)
        np.savetxt(commName + '_dw_cov' + str(noise) + '.txt', dw_cov)
        np.savetxt(commName + '_dtr' + str(noise) + '.txt', pr_dtr)
        np.savetxt(commName + '_dtr_cov' + str(noise) + '.txt', dtr_cov)
        np.savetxt(commName + '_dtr_gnd' + str(noise) + '.txt', pr_dtr_gnd)

def runTrainTest(dsName, subType, seq, seqRange):
    runTrain(dsName, subType, seq)
    runTest(dsName, subType, seqRange)

def runTrain(dsName, subType, seq):
    s = time.time()
    train(dsName, subType, seq)
    print(time.time() - s)

def runTest(dsName, subType, seqRange):
    test(dsName, subType, seqRange)

if __name__ == '__main__':
    dsName = 'airsim'
    seq = [0]
    seqRange = [0, 3]

    runTrainTest('kitti', 'none', seq=[0, 4], seqRange=[0, 11])


# %%
