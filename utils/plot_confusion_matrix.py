import numpy as np
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,save_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        matrix = cm
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()

    figsize = 8, 6
    figure, ax = plt.subplots(figsize=figsize)
    # figure, ax = plt.subplots()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    font_title = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 15,
                  }
    plt.title(title, fontdict=font_title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, )
    plt.yticks(tick_marks, classes)

    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    # print(labels)
    [label.set_fontname('Times New Roman') for label in labels]
    if normalize:
        fm_int = 'd'
        fm_float = '.3f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm_float),
                     horizontalalignment="center", verticalalignment='top', family="Times New Roman",
                     weight="normal", size=15,
                     color="white" if cm[i, j] > thresh else "black")
            plt.text(j, i, format(matrix[i, j], fm_int),
                     horizontalalignment="center", verticalalignment='bottom', family="Times New Roman", weight="normal",
                     size=15,
                     color="white" if cm[i, j] > thresh else "black")
    else:
        fm_int = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm_int),
                     horizontalalignment="center", verticalalignment='bottom',
                     color="white" if cm[i, j] > thresh else "black")



    font_lable = {'family': 'Times New Roman',
                  'weight': 'normal',
                  'size': 15,
                  }
    plt.ylabel('True label', font_lable)
    plt.xlabel('Predicted label', font_lable)
    plt.tight_layout()
    # plt.savefig('../static/imgs/confusion_matrix.eps', dpi=600, format='eps')
    plt.savefig(save_path, dpi=100, format='png')

iemocap_classes = ['Hap&Exc', 'Ang', 'Neu', 'Sad']
meld_classes = ['Neu', 'Sur', 'Joy', 'Sad', 'Ang', 'Fea', 'Dis']
meld7_classes = ['Neu', 'Sur', 'Joy', 'Sad', 'Ang', 'Fea', 'Dis']



# plot_confusion_matrix(iemocap_cnf_matrix, classes=iemocap_classes, normalize=True, title='Normalized Confusion Matrix of IEMOCAP', save_path ='../static/imgs/confusion_matrix/1_confusion_matrix.png')

if __name__ == '__main__':
    iemocap_cnf_matrix = np.array([[243, 4, 11, 20],
                                   [26, 184, 14, 5],
                                   [87, 9, 191, 97],
                                   [11, 4, 11, 168]])

    meld_cnf_matrix = np.array([[409, 2, 0, 1],
                                [0, 560, 2, 0],
                                [1, 0, 422, 1],
                                [9, 1, 1, 180]])

    plot_confusion_matrix(iemocap_cnf_matrix, classes=iemocap_classes, normalize=True, title='Normalized Confusion Matrix of IEMOCAP', save_path ='../static/imgs/confusion_matrix/1_confusion_matrix.png')
