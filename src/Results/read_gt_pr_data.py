import numpy as np
from scipy.integrate import cumulative_trapezoid

def read_pr_data(seq):
    fNamePrefix = 'Data/docker_kitti_none'
    fNameSuffix = '.txt'
    pr_du = np.loadtxt(fNamePrefix + f'{seq}_du0' + fNameSuffix)
    pr_du_cov = np.loadtxt(fNamePrefix + f'{seq}_du_cov0' + fNameSuffix)
    pr_dw = np.loadtxt(fNamePrefix + f'{seq}_dw0' + fNameSuffix)
    pr_dw_cov = np.loadtxt(fNamePrefix + f'{seq}_dw_cov0' + fNameSuffix)
    pr_dtr = np.loadtxt(fNamePrefix + f'{seq}_dtr0' + fNameSuffix)
    pr_dtr_gnd = np.loadtxt(fNamePrefix + f'{seq}_dtr_gnd0' + fNameSuffix)
    pr_dtr_cov = np.loadtxt(fNamePrefix + f'{seq}_dtr_cov0' + fNameSuffix)
    return pr_du, pr_du_cov, pr_dw, pr_dw_cov, pr_dtr, pr_dtr_gnd, pr_dtr_cov

def read_gt_data(seq):
    gt_path = '/workspace/datasets/KITTI/odom/dataset/sequences/'
    gt_path += f'0{seq}/' if seq < 10 else f'{seq}/'
    gt_du = np.loadtxt(f'{gt_path}du.txt', delimiter=',')
    gt_dw = np.loadtxt(f'{gt_path}dw.txt', delimiter=',')
    gt_dtr = np.loadtxt(f'{gt_path}dtrans.txt', delimiter=',')
    gt_dtr_gnd = np.loadtxt(f'{gt_path}dtrans_gnd.txt', delimiter=',')
    gt_pose = np.loadtxt(f'{gt_path}pos.txt', delimiter=',')
    gt_linR = np.loadtxt(f'{gt_path}linR0.txt', delimiter=',')
    return gt_du, gt_dw, gt_dtr, gt_dtr_gnd, gt_pose, gt_linR

def build_covm(data):
    covm = np.zeros((data.shape[0], 3, 3))
    cov3 = np.zeros((data.shape[0], 3))
    for i in range(data.shape[0]):
        l11, l21, l22, l31, l32, l33 = data[i]
        # Construct the lower triangular matrix L
        L = np.array([
            [l11, 0,   0],
            [l21, l22, 0],
            [l31, l32, l33]
        ])
        covm[i] = L @ L.T
        cov3[i] = np.diag(covm[i])
    return covm, cov3

def calc_Q_dTr(gt_data, pred_data):
    Q_dtr_gnd = np.zeros((gt_data.dtr.mu.shape[0], 3, 3))
    cum_Q_dtr_gnd = np.zeros((gt_data.dtr.mu.shape[0], 3, 3))
    dtr_covm = pred_data.dtr.covm
    A = np.eye(3) # This is missing part. TODO: Find the correct A
    for i in range(1, dtr_covm.shape[0]):
        R = gt_data.rotm[i]
        Q_dtr_gnd[i] = R @ dtr_covm[i-1] @ R.T
        cum_Q_dtr_gnd[i] = A @ cum_Q_dtr_gnd[i-1] @ A.T + dtr_covm[i]
    
    return Q_dtr_gnd, cum_Q_dtr_gnd

class PredictionData:
    def __init__(self, seq = 0) -> None:
        pr_du, pr_du_cov, \
        pr_dw, pr_dw_cov, \
        pr_dtr, pr_dtr_gnd, pr_dtr_cov = read_pr_data(seq)
        self.du = EgomotionData(pr_du, pr_du_cov)
        self.dw = EgomotionData(pr_dw, pr_dw_cov)
        self.dtr = EgomotionData(pr_dtr, pr_dtr_cov)
        self.dtr_gnd = pr_dtr_gnd
        self.pose = np.zeros_like(pr_dtr_gnd)
        for i in range(1, len(self.pose)):
            self.pose[i] = self.pose[i-1] + self.dtr_gnd[i-1]
        


class GroundTruthData:
    def __init__(self, seq) -> None:
        gt_du, gt_dw, gt_dtr, gt_dtr_gnd, gt_pose, gt_linR = read_gt_data(seq)
        self.du = EgomotionData(gt_du)
        self.dw = EgomotionData(gt_dw)
        self.dtr = EgomotionData(gt_dtr)
        self.pose = gt_pose
        self.rotm = np.reshape(gt_linR, (gt_linR.shape[0], 3, 3))

class EgomotionData:
    def __init__(self, mu, raw_cov=None) -> None:
        self.mu = mu   # N x 3
        if raw_cov is None:
            return
        covm, cov3 = build_covm(raw_cov)
        self.covm = covm # N x 3 x 3
        self.cov3 = cov3 # N x 3