import torch.nn as nn
import torch
from MyPyTorchAPI.CustomActivation import Sigmoid, TanH



class GuessNet(nn.Module):
    def __init__(self):
        super(GuessNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(1, 100), nn.PReLU(),
                                 nn.Linear(100, 100, nn.PReLU()),
                                 nn.Linear(100, 6), Sigmoid(0.1, 10))

        self.sign = nn.Sequential(nn.Linear(1, 100), nn.Sigmoid(),
                                 nn.Linear(100, 100, nn.Sigmoid()),
                                 nn.Linear(100, 3), TanH(0.1, 1))

    def forward(self):
        x = torch.ones(1,1)
        guess = self.net(x)*-1
        sign = self.sign(x)
        return guess, sign

class TorchKFBLock(nn.Module):
    def __init__(self, gt, dt, prSig, mSig, mCov):
        super().__init__()
        self.gt = torch.from_numpy(gt)
        self.dt = torch.from_numpy(dt)
        self.prSig = torch.from_numpy(prSig)
        self.mSig = torch.from_numpy(mSig)
        self.mCov = torch.from_numpy(mCov)

    def forward(self, guess, sign):
        R = torch.ones(3,3)
        R[0, 0] *= 10 ** guess[0, 0]
        R[1, 1] *= 10 ** guess[0, 1]
        R[2, 2] *= 10 ** guess[0, 2]
        R[1, 0] *= sign[0,0]*10 ** guess[0, 3]
        R[2, 0] *= sign[0,1]*10 ** guess[0, 4]
        R[2, 1] *= sign[0,2]*10 ** guess[0, 5]
        R[0, 1] = R[1, 0]
        R[0, 2] = R[2, 0]
        R[1, 2] = R[2, 1]

        N = self.prSig.shape[0]
        state = torch.zeros(N, 3)
        sysCov = torch.zeros(N,3,3)
        for i in range(1, N):
            # prediction
            prX = state[i-1,:] + self.prSig[i]*self.dt[i]**2 # why dt[i]**2? The frame 2 frame egomition is displacement in meters.
            # The displacement is a change of position. Not position, not velocity. dx/dt = vel, dispalcement = dx = vel*dt
            prCov = sysCov[i-1,:] + R

            # K gain
            K = torch.inverse(prCov + self.mCov[i])
            K = torch.matmul(prCov, K)

            # correction
            innov = self.mSig[i] - prX
            corrX = prX + torch.matmul(K, innov)
            corrCov = prCov - torch.matmul(K, prCov)

            state[i] = corrX
            sysCov[i] = corrCov

        return state

class GetRMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def rmse(self, pr, gt):
        e = pr - gt
        se = e ** 2
        mse = torch.mean(se, dim=0)
        rmse = torch.sqrt(mse)
        return rmse

    def forward(self, kf, gt):
        gt = torch.from_numpy(gt)
        posKF = torch.cumsum(kf, dim=0)
        posGT = torch.cumsum(gt, dim=0)
        posRMSE = self.rmse(posKF, posGT)
        velRMSE = self.rmse(kf, gt)
        return velRMSE, posRMSE