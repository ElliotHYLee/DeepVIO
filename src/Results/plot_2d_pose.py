import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from Results.read_gt_pr_data import PredictionData, GroundTruthData
from read_gt_pr_data import calc_Q_dTr

def plot_covariance_ellipse(x, y, cov, ax, nstd=3, label=None, **kwargs):
    """
    Plots an ellipse representing the covariance matrix at (x, y) point and marks the center with a dot.
    nstd specifies the number of standard deviations the ellipse represents.
    """
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellipse = Ellipse(xy=(x, y), width=width, height=height, angle=theta, fill=False, label=label, **kwargs)
    ax.add_patch(ellipse)
    # Plot a center dot
    ax.plot(x, y, 'ko', markersize=3)  # 'ko' plots a black dot


def plot_2d_pose(gt_pose, pr_pose, cum_Q_dtr_gnd=None):
    fig, ax = plt.subplots()
    ax.plot(gt_pose[:, 0], gt_pose[:, 2], 'ro-', markersize=2, label='Ground Truth')
    ax.plot(pr_pose[:, 0], pr_pose[:, 2], 'b.',  markersize=1, label='Predicted')
    
    if cum_Q_dtr_gnd is not None:
        first_cyan = True
        first_magenta = True
        for i in range(0, 500, 10):
            label = 'Uncertainty (Cyan) - first 500' if first_cyan else None
            plot_covariance_ellipse(pr_pose[i, 0], pr_pose[i, 2], cum_Q_dtr_gnd[i][[0,2]][:,[0,2]], ax, alpha=0.5, color='cyan', label=label)
            first_cyan = False
        for i in range(0, len(pr_pose), 250):
            label = 'Uncertainty (Magenta)' if first_magenta else None
            plot_covariance_ellipse(pr_pose[i, 0], pr_pose[i, 2], cum_Q_dtr_gnd[i][[0,2]][:,[0,2]], ax, alpha=1, color='magenta', label=label)
            first_magenta = False

    ax.set_xlabel('X position')
    ax.set_ylabel('Z position')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    plt.tight_layout(rect=[0, 1, 1, 0.95])  # Adjust layout to make space for legend
    return fig, ax

if __name__ == "__main__":
    seq = 5  # Define KITTI sequence to show
    pred_data = PredictionData(seq)
    gt_data = GroundTruthData(seq)
    gt_pose = gt_data.pose
    pr_pose = pred_data.pose
    Q_dtr_gnd, cum_Q_dtr_gnd = calc_Q_dTr(gt_data, pred_data)
    fig, ax = plot_2d_pose(gt_pose, pr_pose, cum_Q_dtr_gnd)
    plt.show()
