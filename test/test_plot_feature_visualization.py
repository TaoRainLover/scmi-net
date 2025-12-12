#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/12 12:44
# @Author : Tao Zhang
# @Email: 2637050370@qq.com

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


# 这是画一个图
def plot_one(classification_feats, true_y, title):
    # T_sne降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(classification_feats)
    # 定义标签映射
    idx2label = {0: 'Hap&Exc', 1: 'Ang', 2: 'Neu', 3: 'Sad'}
    # 定义颜色映射，您可以根据需要修改颜色
    color_map = {0: '#440154', 1: '#31688e', 2: '#35b779', 3: '#fde726'}
    # 定义形状映射，您可以根据需要修改形状
    marker_map = {0: 'o', 1: 's', 2: '^', 3: 'D'}

    # 可视化 t-SNE 降维结果
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(true_y))

    # 绘制散点图，并使用真实标签作为颜色
    for label in unique_labels:
        indices = [i for i, y in enumerate(true_y) if y == label]
        plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1], marker=marker_map[label], c=color_map[label],
                    label=idx2label[label])

    # 添加图例
    plt.legend()

    # 添加标题
    plt.title(title)

    # 移除坐标值
    plt.xticks([])
    plt.yticks([])
    # 保存图形为PNG格式
    # plt.savefig('feature_visualization.png')
    # 显示图形
    plt.show()


# 将几个图画在一行
def plot_one_line(classification_feats_list, true_y_list, title_list):
    fig, axes = plt.subplots(1, len(classification_feats_list), figsize=(19, 5.05))
    idx2label = {0: 'Hap&Exc', 1: 'Ang', 2: 'Neu', 3: 'Sad'}
    # 定义颜色映射，您可以根据需要修改颜色
    color_map = {0: '#440154', 1: '#31688e', 2: '#35b779', 3: '#fde726'}
    # 定义形状映射，您可以根据需要修改形状
    marker_map = {0: 'o', 1: 's', 2: '^', 3: 'D'}
    for index in range(len(classification_feats_list)):
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(classification_feats_list[index])
        unique_labels = list(set(true_y_list[index]))
        # print('unique_labels: {}'.format(unique_labels))
        # 绘制散点图，并使用真实标签作为颜色
        for label in unique_labels:
            indices = [i for i, y in enumerate(true_y_list[index]) if y == label]
            axes[index].scatter(features_tsne[indices, 0], features_tsne[indices, 1], marker=marker_map[label],
                                c=color_map[label], label=idx2label[label], s=15)

        # 设置标题
        axes[index].set_title(title_list[index], fontsize=22)
        # 移除坐标值
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        # axes[index].set_aspect('equal')

    # 调整子图之间的间距
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.02)  # 水平方向的间隙大小
    # 添加图例
    # plt.legend()
    # 添加图例，并放到图的外面横向展示
    plt.legend(loc='upper center', bbox_to_anchor=(-1, -0.005), fancybox=True, shadow=True, ncol=len(idx2label),
               fontsize=16)
    # 保存图形为PNG格式
    # plt.savefig(args.model_path+str(args.session_id)+'_feature_visualization.png')
    plt.show()


if __name__ == '__main__':
    # 读取SCMI-Net的特征集
    npy_file_path4 = '../results/iemocap/20240227-SCMI模型特征可视化分析/2_classification_feats.npy'
    npy_file_path3 = '../results/iemocap/20240229-baseline特征可视化分析/2_classification_feats.npy'
    npy_file_path1 = '../results/iemocap/20240301-语音模态可视化分析/2_classification_feats.npy'
    npy_file_path2 = '../results/iemocap/20240301-文本模态特征可视化分析/2_classification_feats.npy'

    true_y_file_path4 = '../results/iemocap/20240227-SCMI模型特征可视化分析/2_true_y.npy'
    true_y_file_path3 = '../results/iemocap/20240229-baseline特征可视化分析/2_true_y.npy'
    true_y_file_path1 = '../results/iemocap/20240301-语音模态可视化分析/2_true_y.npy'
    true_y_file_path2 = '../results/iemocap/20240301-文本模态特征可视化分析/2_true_y.npy'

    title4, title3, title1, title2 = 'SCMI-Net', 'Baseline', 'Audio', 'Text',

    npy_file_path_list = [npy_file_path1, npy_file_path2, npy_file_path3, npy_file_path4]
    true_y_file_list = [true_y_file_path1, true_y_file_path2, true_y_file_path3, true_y_file_path4]
    title_list = [title1, title2, title3, title4]

    classification_feats_list, true_y_list = [], []

    for feat_npy_file_path, ture_y_npy_path in zip(npy_file_path_list, true_y_file_list):
        classification_feats_list.append(np.load(feat_npy_file_path))
        true_y_list.append(np.load(ture_y_npy_path))

    # TODO：测试_可视化_画单个图
    plot_one(classification_feats=classification_feats_list[0], true_y=true_y_list[0], title="Audio")
    # TODO: 测试_可视化_将几个图同时画在一行
    # plot_one_line(classification_feats_list, true_y_list, title_list)
