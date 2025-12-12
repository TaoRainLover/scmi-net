import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def get_preds(self, preds):
    """
    Get predictions for all utterances from their segments' prediction.
    This function will accumulate the predictions for each utterance by
    taking the maximum probability along the dimension 0 of all segments
    belonging to that particular utterance.
    """
    preds = np.argmax(preds, axis=1)
    return preds

def unweighted_accuracy(label_true, label_pred):
    """
    Calculate accuracy score given the predictions.

    Parameters
    ----------
    predictions : ndarray
        Model's predictions.

    Returns
    -------
    float
        Accuracy score.

    """
    label_true = np.array(label_true)
    label_pred = np.array(label_pred)
    acc = (label_true == label_pred).sum() / len(label_true)
    return acc


def weighted_accuracy(label_true, label_pred, num_classes = 4):
    """
    Calculate unweighted accuracy score given the predictions.

    Parameters
    ----------
    utt_preds : ndarray
        Processed predictions.

    Returns
    -------
    float
        Unweighted Accuracy (UA) score.

    """

    class_acc = 0
    n_classes = 0
    label_true = np.array(label_true)
    label_pred = np.array(label_pred)
    for c in range(num_classes):
        class_pred = np.multiply((label_true == label_pred),
                                 (label_true == c)).sum()

        if (label_true == c).sum() > 0:
            class_pred /= (label_true == c).sum()
            n_classes += 1
            class_acc += class_pred

    return class_acc / n_classes


def confusion_matrix_iemocap(label_true, label_pred):
    label_true = np.array(label_true)
    label_pred = np.array(label_pred)
    """Compute confusion matrix given the predictions.

    Parameters
    ----------
    utt_preds : ndarray
        Processed predictions.

    """
    conf = confusion_matrix(label_true, label_pred)

    # Make confusion matrix into data frame for readability
    conf_fmt = pd.DataFrame({"hap": conf[:, 0], "ang": conf[:, 1],
                             "neu": conf[:, 2], "sad": conf[:, 3]})
    conf_fmt = conf_fmt.to_string(index=False)
    print(conf_fmt)
    return (conf, conf_fmt)

def confusion_matrix_meld(label_true, label_pred, num_labels = 7):
    label_true = np.array(label_true)
    label_pred = np.array(label_pred)
    """Compute confusion matrix given the predictions.

    Parameters
    ----------
    utt_preds : ndarray
        Processed predictions.

    """
    conf = confusion_matrix(label_true, label_pred)

    # Make confusion matrix into data frame for readability
    if num_labels == 7:
        conf_fmt = pd.DataFrame({"class1": conf[:, 0], "class2": conf[:, 1],
                             "class3": conf[:, 2], "class4": conf[:, 3],
                             "class5": conf[:, 4], "Fea": conf[:, 5],
                             "Dis": conf[:, 6]})
    elif num_labels == 5:
        conf_fmt = pd.DataFrame({"class1": conf[:, 0], "class2": conf[:, 1],
                                 "class3": conf[:, 2], "class4": conf[:, 3],
                                 "class5": conf[:, 4]})
    conf_fmt = conf_fmt.to_string(index=False)
    print(conf_fmt)
    return (conf, conf_fmt)