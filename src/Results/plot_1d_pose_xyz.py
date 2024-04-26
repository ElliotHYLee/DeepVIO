import numpy as np
import matplotlib.pyplot as plt
from Results.read_gt_pr_data import PredictionData, GroundTruthData
from read_gt_pr_data import calc_Q_dTr

def plot_1d_pose_xyz(gt_pose, pr_pose, cum_Q_dtr_gnd=None):
    
    # 1D Pose XYZ with uncertainty
    plt.figure()

    # x
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(gt_pose[:, 0], 'ro-', markersize=2, label='Ground Truth')
    ax1.plot(pr_pose[:, 0], 'b.',  markersize=1, label='Predicted')
    if cum_Q_dtr_gnd is not None:
        for i in range(0, len(pr_pose), 20):
            std3 = np.sqrt(np.diag(cum_Q_dtr_gnd[i]))
            ax1.plot([i, i], [pr_pose[i, 0] - 6*std3[0], pr_pose[i, 0] + 6*std3[0]], '-', color="cyan", alpha=0.5, linewidth=1)
    ax1.set_ylabel('X (m)')
    ax1.set_xlabel('Frame Index')

    # y
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(gt_pose[:, 1], 'ro-', markersize=2, label='Ground Truth')
    ax2.plot(pr_pose[:, 1], 'b.',  markersize=1, label='Predicted')
    if cum_Q_dtr_gnd is not None:
        for i in range(0, len(pr_pose), 20):
            std3 = np.sqrt(np.diag(cum_Q_dtr_gnd[i]))
            ax2.plot([i, i], [pr_pose[i, 1] - 6*std3[1], pr_pose[i, 1] + 6*std3[1]], '-', color="cyan", alpha=0.5, linewidth=1)
    ax2.set_ylabel('Y (m)')
    ax2.set_xlabel('Frame Index')

    # z
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(gt_pose[:, 2], 'ro-', markersize=2, label='Ground Truth')
    ax3.plot(pr_pose[:, 2], 'b.',  markersize=1, label='Predicted')
    if cum_Q_dtr_gnd is not None:
        for i in range(0, len(pr_pose), 20):
            std3 = np.sqrt(np.diag(cum_Q_dtr_gnd[i]))
            ax3.plot([i, i], [pr_pose[i, 2] - 6*std3[2], pr_pose[i, 2] + 6*std3[2]], '-', color="cyan", alpha=0.5, linewidth=1, label='Uncertainty' if i == 0 else "")
    ax3.set_ylabel('Z (m)')
    ax3.set_xlabel('Frame Index')

    # Adding an overall legend
    handles, labels = ax1.get_legend_handles_labels()
    # Manually add the last handle and label for the "Uncertainty"
    last_handle, last_label = ax3.get_legend_handles_labels()
    handles.append(last_handle[-1])
    labels.append(last_label[-1])
    plt.figlegend(handles, labels, loc='upper center', ncol=3, labelspacing=0.)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 

if __name__ == "__main__":
    # define KITTI sequence to show
    seq = 0
    # read data
    pred_data = PredictionData(seq)
    gt_data = GroundTruthData(seq)
    gt_pose = gt_data.pose
    pr_pose = pred_data.pose
    
    Q_dtr_gnd, cum_Q_dtr_gnd = calc_Q_dTr(gt_data, pred_data)
    # plot_2d_pose
    plot_1d_pose_xyz(gt_pose, pr_pose, cum_Q_dtr_gnd)
    plt.show()

