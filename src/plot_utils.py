import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

font_legend = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 16
    }
font_title = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 22
}

def plot_one_line(classification_feats_list, true_y_list, title_list):
    fig, axes = plt.subplots(1, len(classification_feats_list), figsize=(19, 5.05))
    idx2label = ['Hap&Exc', 'Ang', 'Neu', 'Sad']
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
        # axes[index].set_title(title_list[index], fontsize=22, fontdict=font_title)
        axes[index].set_title(title_list[index], fontsize=22)
        # 移除坐标值
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        # axes[index].set_aspect('equal')

    # 调整子图之间的间距
    # plt.tight_layout()A
    plt.subplots_adjust(wspace=0.02)  # 水平方向的间隙大小
    # 添加图例
    # plt.legend()
    # 添加图例，并放到图的外面横向展示
    # plt.legend(loc='upper center', bbox_to_anchor=(-1, -0.005), fancybox=True, shadow=True, ncol=len(idx2label), fontsize=16,  prop=font_legend)
    plt.legend(loc='upper center', bbox_to_anchor=(-1, -0.005), fancybox=True, shadow=True, ncol=len(idx2label), fontsize=16)
    # 保存图形为PNG格式
    # plt.savefig(args.model_path+str(args.session_id)+'_feature_visualization.png')
    plt.show()


def plot_nxn(classification_feats_list, true_y_list, title_list):
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))  # 2行2列的子图

    idx2label = ['Hap&Exc', 'Ang', 'Neu', 'Sad']
    color_map = {0: '#440154', 1: '#31688e', 2: '#35b779', 3: '#fde726'}
    marker_map = {0: 'o', 1: 's', 2: '^', 3: 'D'}

    for index, ax in enumerate(axes.flatten()):  # 对所有子图进行循环
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(classification_feats_list[index])
        unique_labels = list(set(true_y_list[index]))

        for label in unique_labels:
            indices = [i for i, y in enumerate(true_y_list[index]) if y == label]
            ax.scatter(features_tsne[indices, 0], features_tsne[indices, 1], marker=marker_map[label],
                       c=color_map[label], label=idx2label[label], s=15)

        ax.set_title(title_list[index])
        ax.set_xticks([])
        ax.set_yticks([])

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.05, hspace=0.15)  # 调整水平和垂直方向的间距大小

    # 添加图例，并放到图的外面横向展示
    plt.legend(loc='center left', bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=1)

    plt.show()


if __name__ == '__main__':
    # 读取SCMI-Net的特征集
    npy_file_path4 = '../results/iemocap/20240227-SCMI模型特征可视化分析/2_classification_feats.npy'
    npy_file_path3 = '../results/iemocap/20240229-baseline特征可视化分析/2_classification_feats.npy'
    npy_file_path1 = '../results/iemocap/20240301-语音模态可视化分析/2_classification_feats.npy'
    npy_file_path2 = '../results/iemocap/20240301-文本模态特征可视化分析/2_classification_feats.npy'

    true_y_file_path4= '../results/iemocap/20240227-SCMI模型特征可视化分析/2_true_y.npy'
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

    # 绘图
    plot_one_line(classification_feats_list, true_y_list, title_list)
