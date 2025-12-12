#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/1/22 22:04
# @Author : Tao Zhang
# @Email: 2637050370@qq.com

import numpy as np
import itertools
import matplotlib.pyplot as plt


# 绘制混淆矩阵
import numpy as np
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(matrix_list, classes_list, save_path, normalize=False, title_list=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Inputs:
    - matrix_list: A list of confusion matrices
    - classes_list: List of lists containing class labels for each matrix
    - save_path: File path to save the plot
    - normalize: True to show percentages, False to show counts
    - title_list: List of titles for each matrix
    - cmap: Colormap to be used in the plot
    """
    # 设置横纵坐标的名称以及对应字体格式
    font_lable = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 15,
                  }
    # 设置title的大小以及title的字体
    font_title = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 15,
                  }

    if normalize:
        cm_list = matrix_list
        matrix_list = [cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] for cm in matrix_list]

    num_matrices = len(matrix_list)
    rows = 1
    cols = num_matrices

    fig, axes = plt.subplots(rows, cols, figsize=(16, 6))

    for i, (cm, classes) in enumerate(zip(matrix_list, classes_list)):
        # ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax = axes[i]
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title_list[i] if title_list else f'Matrix {i + 1}', font_title)
        ax.set_xlabel('Predicted label', font_lable)
        ax.set_ylabel('True label', font_lable)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.tick_params(labelsize=12)
        ax.set_xticklabels(classes, rotation=45, fontname='Times New Roman')
        ax.set_yticklabels(classes, fontname='Times New Roman')
        # ax.tick_params(axis='both', which='major', labelsize=12, width=1, length=5, direction='inout')

        if normalize:
            fm_int = 'd'
            fm_float = '.3f'
            thresh = cm.max() / 2.
            for i_2, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                ax.text(j, i_2, format(cm[i_2, j], fm_float),
                        horizontalalignment="center", verticalalignment='top',
                        family="Times New Roman",  weight="normal", size=15,
                        color="white" if cm[i_2, j] > thresh else "black")
                ax.text(j, i_2, format(cm_list[i][i_2, j], fm_int),
                        horizontalalignment="center", verticalalignment='bottom',
                        family="Times New Roman", weight="normal", size=15,
                        color="white" if cm[i_2, j] > thresh else "black")
        else:
            fm_int = 'd'
            thresh = cm.max() / 2.
            for i_2, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                ax.text(j, i_2, format(cm[i_2, j], fm_int),
                        horizontalalignment="center", verticalalignment='bottom',
                        color="white" if cm[i_2, j] > thresh else "black", fontname='Times New Roman')

        fig.colorbar(im, ax=ax)

    axes[1].set_xlabel('Predicted label', font_lable, labelpad=24)

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.02)  # 水平方向的间隙大小
    # plt.tight_layout()
    plt.savefig(save_path, dpi=100, format='png')
    plt.show()


# Example usage:
# Assuming you have a list of confusion matrices named matrix_list
# a list of class labels named classes_list
# and a list of titles named title_list
# plot_confusion_matrix(matrix_list, classes_list, 'output.png', normalize=True, title_list=title_list)


iemocap_classes = ['Hap&Exc', 'Ang', 'Neu', 'Sad']
meld5_classes = ['Neu', 'Sur', 'Joy', 'Sad', 'Ang']
meld7_classes = ['Neu', 'Sur', 'Joy', 'Sad', 'Ang', 'Fea', 'Dis']



# plot_confusion_matrix(iemocap_cnf_matrix, classes=iemocap_classes, normalize=True, title='Normalized Confusion Matrix of IEMOCAP', save_path ='../static/imgs/confusion_matrix/1_confusion_matrix.png')

if __name__ == '__main__':
    iemocap_cnf_matrix = np.array([[243, 4, 11, 20],
                                   [26, 184, 14, 5],
                                   [87, 9, 191, 97],
                                   [11, 4, 11, 168]])

    meld_cnf_matrix = np.array([[1111, 25, 66, 24, 30],
                                [70, 145, 38, 5, 23],
                                [112, 19, 244, 7, 20],
                                [118, 12, 15, 41, 22],
                                [112, 37, 52, 4, 140],])

    matrix_list = [iemocap_cnf_matrix, meld_cnf_matrix]
    title_list = ['Normalized Confusion Matrix of IEMOCAP', 'Normalized Confusion Matrix of MELD']
    classes_list = [iemocap_classes, meld5_classes]

    # plot_confusion_matrix([iemocap_cnf_matrix, iemocap_cnf_matrix], classes=iemocap_classes, normalize=True, save_path ='./confusion_matrix.png')
    plot_confusion_matrix(matrix_list, classes_list=classes_list, title_list=title_list, normalize=True, save_path ='../static/imgs/confusion_matrix.png')
