import numpy as np
import matplotlib.pyplot as plt
from Results.read_gt_pr_data import PredictionData, GroundTruthData

def plot_data(gt, pr, title):
    if gt is not None:
        plt.plot(gt, 'ro-', label='Ground Truth', )
    plt.plot(pr, 'b.', markersize=0.5, label='Predicted')
    plt.grid()
    plt.title(title)

def plot_vect3(gt, pr, title, y_label='Value'):

    plt.figure(figsize=(8, 4))
    plt.suptitle(title, fontsize=16)
    ##================================================================================================
    ## mu
    ##================================================================================================
    # Plot mu x
    plt.subplot(2, 3, 1)
    plot_data(gt.mu[:, 0], pr.mu[:, 0], f'X')
    plt.xlabel('Frame Index')
    plt.ylabel(y_label)
    

    # Plot mu y 
    plt.subplot(2, 3, 2)
    plot_data(gt.mu[:, 1], pr.mu[:, 1], f'Y')
    plt.xlabel('Frame Index')
    plt.ylabel(y_label)

    # Plot mu z
    plt.subplot(2, 3, 3)
    plot_data(gt.mu[:, 2], pr.mu[:, 2], f'Z')
    plt.xlabel('Frame Index')
    plt.ylabel(y_label)
    plt.legend(loc='upper center', bbox_to_anchor=(0.8, 1.9))

    ##================================================================================================
    ## cov
    ##================================================================================================
    plt.subplot(2, 3, 4)
    plot_data(None, pr.cov3[:, 0], f'Covariance - X')
    plt.xlabel('Frame Index')
    plt.ylabel(r'1$\sigma$_' + y_label)
    plt.subplot(2, 3, 5)
    plot_data(None, pr.cov3[:, 1], f'Covariance - Y')
    plt.xlabel('Frame Index')
    plt.ylabel(r'1$\sigma$_' + y_label)
    plt.subplot(2, 3, 6)
    plot_data(None, pr.cov3[:, 2], f'Covariance - Z')
    plt.xlabel('Frame Index')
    plt.ylabel(r'1$\sigma$_' + y_label)
    plt.subplots_adjust(hspace=1, wspace=0.5, top=0.8) 

def plot_basic(gt_data, pred_data):
    # plot basic data
    plot_vect3(gt_data.du, pred_data.du, 'Linear Motion, du (m)', r'$\Delta u$' )
    plot_vect3(gt_data.dw, pred_data.dw, 'Rotational Motion, dw (m)', r'$\Delta w$' )
    plot_vect3(gt_data.dtr, pred_data.dtr, 'Translational Motion, dw (m)', r'$\Delta tr$' )

if __name__ == "__main__":
    # define KITTI sequence to show
    seq = 0
    # read data
    pred_data = PredictionData(seq)
    gt_data = GroundTruthData(seq)
    # plot
    plot_basic(seq, gt_data, pred_data)
    # show
    plt.show()