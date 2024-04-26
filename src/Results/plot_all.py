import numpy as np
import matplotlib.pyplot as plt
from Results.plot_1d_pose_xyz import plot_1d_pose_xyz
from Results.plot_2d_pose import plot_2d_pose
from read_gt_pr_data import calc_Q_dTr
from Results.read_gt_pr_data import PredictionData, GroundTruthData
from Results.plot_basic_data import plot_basic


seq = 7
pred_data = PredictionData(seq)
gt_data = GroundTruthData(seq)

gt_pose = gt_data.pose
pr_pose = pred_data.pose

Q_dtr_gnd, cum_Q_dtr_gnd = calc_Q_dTr(gt_data, pred_data)

plot_basic(gt_data, pred_data)
plot_1d_pose_xyz(gt_pose, pr_pose, cum_Q_dtr_gnd)
plot_2d_pose(gt_pose, pr_pose, cum_Q_dtr_gnd)


plt.show()








