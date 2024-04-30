from src.DataReader.CNN_Data.VODataSet import VODataSetManager_CNN
from src.DataReader.KF_Data.KF_PrepData import DataManager
from src.Models.CNN_Model.Model_CNN_0 import Model_CNN_0
from src.Models.CNN_Model.CNN_ModelContainer import CNN_ModelContainer
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.Models.KF_Model.KF_Model import GuessNet
from src.Params import *
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

GPU_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reshapeImage(img):
    return np.reshape(img, (360, 720, 3))

def get_input_data(dataInTuple):
    img0, img1,\
    du, dw, dw_gyro, dw_gyro_stand, \
    dtr, dtr_gnd, rotM = dataInTuple

    img0 = img0.to(GPU_DEVICE)  # image t-1
    img1 = img1.to(GPU_DEVICE)  # image t
    dw = dw.to(GPU_DEVICE)      # gt_angvel
    dw_gyro = dw_gyro.to(GPU_DEVICE) # gyroscop_angvel
    dw_gyro_stand = dw_gyro_stand.to(GPU_DEVICE) # gyroscope noise
    rotM = rotM.to(GPU_DEVICE)      # rotation matrix by IMU. Note: in this project, it is derived from GT!

    # return only the data that is needed for the model input
    return img0, img1, dw_gyro, dw_gyro_stand, rotM

def construct_covm(pr_dtr_chol):
    L11 = pr_dtr_chol[0, 0]
    L21 = pr_dtr_chol[0, 1]
    L22 = pr_dtr_chol[0, 2]
    L31 = pr_dtr_chol[0, 3]
    L32 = pr_dtr_chol[0, 4]
    L33 = pr_dtr_chol[0, 5]
    L = np.array([[L11, 0, 0], 
                  [L21, L22, 0],
                  [L31, L32, L33]])
    return np.matmul(L, L.T)

def prep_acc_data(seq):
    dm = DataManager()
    dm.initHelper(dsName='kitti', subType='none', seq=[seq])
    dt = dm.dt
    acc_gnd = dm.acc_gnd # accleration * dt in ground frame (NOT GROUND TRUTH!)
    return dt, acc_gnd

def setR(guess, sign): # construct process noise of Acc.
    R = np.ones((3,3), dtype=np.float32)
    R[0, 0] *= 10 ** guess[0, 0]
    R[1, 1] *= 10 ** guess[0, 1]
    R[2, 2] *= 10 ** guess[0, 2]
    R[1, 0] *= sign[0, 0]  * 10 ** guess[0, 3]
    R[2, 0] *= sign[0, 1]  * 10 ** guess[0, 4]
    R[2, 1] *= sign[0, 2]  * 10 ** guess[0, 5]
    R[0, 1] = R[1, 0]
    R[0, 2] = R[2, 0]
    R[1, 2] = R[2, 1]
    return R

def test(dsName, subType, seq):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    
    # prepare data
    dm = VODataSetManager_CNN(dsName=dsName, subType=subType, seq=[seq], isTrain=False)
    mc = CNN_ModelContainer(Model_CNN_0(dsName), wName=wName)
    mc.load_weights(wName +'_best', train=False)

    # model in GPU
    the_test_data = dm.testSet
    the_model = mc.model
    dataLoader = DataLoader(the_test_data, batch_size=1, shuffle=False) # 1 data point at a time for a real-time setup

    # Guess Net
    gnet = GuessNet()
    checkPoint = torch.load(f"{wName}_KF.pt")
    gnet.load_state_dict(checkPoint['model_state_dict'])
    gnet.load_state_dict(checkPoint['optimizer_state_dict'])
    guess, sign = gnet()
    params = guess.data.numpy()
    paramsSign = sign.data.numpy()

    # KF
    proc_noise = setR(params, paramsSign) # process noise of Acc.
    dt, acc_gnd = prep_acc_data(seq)
    N = acc_gnd.shape[0]
    
    est_vel = np.zeros((N, 3))
    est_vel_cov = np.zeros((N, 3, 3))
    est_pos = np.zeros((N, 3))
    
    current_vel = np.zeros(3)
    current_vel_cov = np.eye(3)
    current_pos = np.zeros(3)

    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    for batch_idx, dataInTuple in enumerate(dataLoader):
        tic = time.time()
        ##==================================================================================================
        ## In GPU
        ##==================================================================================================
        img0, img1, dw_gyro, dw_gyro_stand, rotM = get_input_data(dataInTuple)

        # Note: bad name warning
        # The variables with *_cov is the Cholesky factor of the covariance matrix. Not the covariance matrix itself.
        pr_du, pr_du_cov, \
        pr_dw, pr_dw_cov, \
        pr_dtr, pr_dtr_cov, \
        pr_dtr_gnd  = the_model(img0, img1, dw_gyro, dw_gyro_stand, rotM)
        ##==================================================================================================
        
        # prep data for KF back in CPU
        cpu_img0 = reshapeImage(img0.cpu().data.numpy()) # 360x720x3
        cpu_img1 = reshapeImage(img1.cpu().data.numpy()) # 360x720x3
        cpu_pr_dtr_gnd = pr_dtr_gnd.cpu().data.numpy()   # 1x3: translation in ground frame (NOT GROUND TRUTH!)
        cpu_pr_dtr_cov = pr_dtr_cov.cpu().data.numpy()   # 1x6: Cholesky factor of the covariance matrix
        cpu_pr_dtr_cov_gnd = construct_covm(cpu_pr_dtr_cov) # 3x3: covariance matrix in ground frame
        
        # KF - prediction step       
        prX = current_vel + acc_gnd[batch_idx]*dt[batch_idx]**2 # why dt^2? because I made a mistake in data_gen by dividing vel by dt^2. TODO: fix data gen @ make_trainable_data.m
        prCov = current_vel_cov + proc_noise

        # KF - correction step
        K = np.linalg.inv(prCov + cpu_pr_dtr_cov_gnd) # Kalman gain f(cov.of acc and cov.of dtr)
        K = np.matmul(prCov, K)
        innov = (cpu_pr_dtr_gnd - prX).T
        corrX = prX + (np.matmul(K, innov)).T
        corrCov = prCov - np.matmul(K, prCov)

        # update data
        current_vel = corrX
        current_vel_cov = corrCov

        est_vel[batch_idx] = current_vel
        est_vel_cov[batch_idx] = current_vel_cov
        current_pos = current_pos + corrX # why no dt? dtr has dtr has in it
        est_pos[batch_idx] = current_pos

        #===============================================
        # Uncomment below for plotting (slows down the process)
        # fig.clear()
        # dtrX = fig.add_subplot(2,2,1)
        # dtrY = fig.add_subplot(2,2,2)
        # dtrZ = fig.add_subplot(2,2,3)
        # pos2D = fig.add_subplot(2,2,4)
        
        # dtrX.plot(est_vel[:batch_idx, 0], 'b-')
        # dtrX.set_ylabel('dTransX, (m)')
        # dtrY.plot(est_vel[:batch_idx, 1], 'b-')
        # dtrY.set_ylabel('dTransY, (m)')
        # dtrZ.plot(est_vel[:batch_idx, 2], 'b-')        
        # dtrZ.set_ylabel('dTransZ, (m)')
        # pos2D.plot(est_pos[:batch_idx,0], est_pos[:batch_idx, 2], 'b.')
        # pos2D.set_ylabel('Position Z, (m)')
        # pos2D.set_xlabel('Position X, (m)')
        # plt.tight_layout()
        # plt.subplots_adjust(wspace=0.5)
        # plt.pause(0.0001)
        # fig.canvas.draw()
        #===============================================

        cv2.imshow('img0', cpu_img0)
        cv2.imshow('img1', cpu_img1)

        # time.sleep(5)
        key = cv2.waitKey(1)
        if key == 27:
            break

        toc = time.time()
        print(f"batch: {batch_idx}, dt: {toc-tic}, freq: {1/(toc-tic)} Hz")

    plt.figure()
    plt.subplot(311)
    plt.plot(est_vel[:, 0], 'r.')
    plt.subplot(312)
    plt.plot(est_vel[:, 1], 'r.')
    plt.subplot(313)
    plt.plot(est_vel[:, 2], 'r.')

    plt.figure()
    plt.plot(est_pos[:, 0], est_pos[:, 2], 'r.')
    
    plt.show()




if __name__ == '__main__':
    test('kitti', 'none', seq=5)

