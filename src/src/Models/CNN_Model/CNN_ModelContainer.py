from MyPyTorchAPI.AbsModelContainer import *
from MyPyTorchAPI.CustomLoss import MahalanobisLoss
import torch.optim as optim

class CNN_ModelContainer(AbsModelContainer):
    def __init__(self, model, wName='Weights/main'):
        super().__init__(model, wName)
        self.bn = 0
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=10 ** -3, weight_decay=10 ** -2)
        self.loss = MahalanobisLoss(isSeries=False)

    def changeOptim(self, epoch):
        if epoch == 3:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=10 ** -4, weight_decay=10 ** -4)

    def forwardProp(self, dataInTuple):
        #pass
        # (x, y, u) = dataInTuple
        img0, img1,\
        du, dw, dw_gyro, dw_gyro_stand, \
        dtr, dtr_gnd, rotM = dataInTuple

         # self.x, self.y, self.u = self.toGPU(x, y, u)
        self.img0 = img0.to(self.device)
        self.img1 = img1.to(self.device)
        self.du = du.to(self.device)
        self.dw = dw.to(self.device)
        self.dw_gyro = dw_gyro.to(self.device)
        self.dw_gyro_stand = dw_gyro_stand.to(self.device)
        self.dtr = dtr.to(self.device)
        self.dtr_gnd = dtr_gnd.to(self.device)
        self.rotM = rotM.to(self.device)

        # Note: bad name warning - variables with *_cov is the Cholesky factor of the covariance matrix. Not the covariance matrix itself.
        self.pr_du, self.pr_du_cov, \
        self.pr_dw, self.pr_dw_cov, \
        self.pr_dtr, self.pr_dtr_cov, \
        self.pr_dtr_gnd = self.model(img0, img1, dw_gyro, dw_gyro_stand, rotM)

    def getLoss(self):
        batch_loss = self.loss(self.pr_du, self.du, self.pr_du_cov) + \
                     self.loss(self.pr_dw, self.dw, self.pr_dw_cov) + \
                     self.loss(self.pr_dtr, self.dtr, self.pr_dtr_cov) + \
                     self.loss(self.pr_dtr_gnd, self.dtr_gnd, self.pr_dtr_cov, self.rotM)
        return batch_loss

    def prepResults(self, N):
        self.result0 = np.zeros((N, self.pr_du.shape[1]))
        self.result1 = np.zeros((N, self.pr_du_cov.shape[1]))
        self.result2 = np.zeros((N, self.pr_dw.shape[1]))
        self.result3 = np.zeros((N, self.pr_dw_cov.shape[1]))
        self.result4 = np.zeros((N, self.pr_dtr.shape[1]))
        self.result5 = np.zeros((N, self.pr_dtr_cov.shape[1]))
        self.result6 = np.zeros((N, self.pr_dtr_gnd.shape[1]))

    def saveToResults(self, start, last):
        self.result0[start:last, :] = self.toCPUNumpy(self.pr_du)
        self.result1[start:last, :] = self.toCPUNumpy(self.pr_du_cov)
        self.result2[start:last, :] = self.toCPUNumpy(self.pr_dw)
        self.result3[start:last, :] = self.toCPUNumpy(self.pr_dw_cov)
        self.result4[start:last, :] = self.toCPUNumpy(self.pr_dtr)
        self.result5[start:last, :] = self.toCPUNumpy(self.pr_dtr_cov)
        self.result6[start:last, :] = self.toCPUNumpy(self.pr_dtr_gnd)

    def returnResults(self):
        return self.result0, self.result1, \
               self.result2, self.result3, \
               self.result4, self.result5, \
               self.result6









