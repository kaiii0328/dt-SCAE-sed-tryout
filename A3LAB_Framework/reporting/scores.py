import os, sys
import matplotlib
try:
    os.environ['DISPLAY']
except:
    if sys.platform == 'linux':
        # if a screen is unavailable
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, classification_report, f1_score, \
    precision_recall_curve, average_precision_score, roc_auc_score
import logging
import time
import numpy as np
import pandas as pd
import seaborn

# import itertools


def compute_score(predictions, labels, verbose=0, numeric_labels=[0.0, 0.1], roc_auc=None):
    """
    :param predictions: list of predicted classes
    :param labels: list of true labels
    :param verbose: verbosity - 1 for synthetic, 2 for full report
    :return: fold_metrics = dictionary with scores
    """
    fold_metrics = {
        'UAR': 0,
        'Accuracy': 0,
        'F1_Report': 0,
        'ConfMatrix': None,
        'Average Precision': None,
    }

    A = accuracy_score(labels, predictions)
    f1_report = f1_score(labels, predictions, labels=numeric_labels, average=None)
    UAR = recall_score(labels, predictions, average='macro')
    CM = confusion_matrix(labels, predictions)
    # compute average prec

    if verbose == 1 or verbose == 2:
        logging.info("Accuracy (%) = " + "{0:.2f}".format(A * 100))
        logging.info("UAR (%) = " + "{0:.2f}".format(UAR * 100))
        logging.info("F1 Score (SNORE) (%) = " + "{0:.2f}".format(f1_report[1] * 100))
        logging.info("AUC_ROC = " + "{0:.2f}".format(roc_auc * 100))
        if verbose == 2:
            cm = CM.astype(int)
            class_report = classification_report(labels, predictions, target_names=['NS', 'S'])
            logging.info("FINAL REPORT")
            logging.info("\t NS\t S")
            logging.info("NS \t" + str(cm[0, 0]) + "\t" + str(cm[0, 1]))
            logging.info("S \t" + str(cm[1, 0]) + "\t" + str(cm[1, 1]))
            logging.info(class_report)
    fold_metrics['Accuracy'] = A
    fold_metrics['UAR'] = UAR
    fold_metrics['F1_Report'] = f1_report
    fold_metrics['ConfMatrix'] = CM
    fold_metrics['ROC'] = roc_auc

    return fold_metrics


def print_scores(result_dict, experiment_time, paths):

    accuracy = result_dict['average']['fold_based']['Accuracy']
    uar = result_dict['average']['fold_based']['UAR']
    f1_sn = result_dict['average']['fold_based']['F1_sn']
    f1_ns = result_dict['average']['fold_based']['F1_ns']
    roc = result_dict['average']['fold_based']['ROC']

    scores_file_path = os.path.join(paths['root_path'], 'Scores_Report.csv')
    lock_file_path = os.path.join(paths['root_path'], '.Scores_Report.lock')
    try:
        logging.info("open File to lock")
        if lock_file(lock_file_path):
            file_to_lock = open(scores_file_path, 'a+')
            if os.path.isfile(scores_file_path) and os.path.getsize(scores_file_path) == 0:
                # scores_string = "FIRST LINE OF RESULTS CSV"
                scores_string = "    {process_id:<25s} | {accuracy:<15s} | {UAR:<15s} |" \
                                " {f1_sn:<20s} | {roc:<15s} | {Time:<40s} \n".format(
                                    process_id='Process_id',
                                    accuracy="Accuracy (%)",
                                    UAR="UAR (%)",
                                    f1_sn="F1 SNORE (%)",
                                    roc="ROC (%)",
                                    Time="Experiment Time (DAYS:HOURS:MIN:SEC)")

                file_to_lock.write(scores_string)
            unlock_file(lock_file_path)
    except OSError as exception:
        logging.info(exception)
        raise
    # prova a bloccare il file: se non riesce ritenta dopo un po. Non va avanti finche non riesce a bloccare il file
    try:
        while True:

            if lock_file(lock_file_path):
                break
            else:
                logging.info("wait fo file to Lock")
                time.sleep(0.1)

        logging.info("Print Results on Scores Report")

        scores_string = "    {process_id:<25s} | {accuracy:<15s} | {UAR:<15s} |" \
                        " {f1_sn:<20s} | {roc:<15s} | {Time:<40s} \n".format(
                            process_id=paths['string_id'],
                            accuracy="{0:.2f}".format(accuracy * 100),
                            UAR="{0:.2f}".format(uar * 100),
                            f1_sn="{0:.2f}".format(f1_sn * 100),
                            roc="{0:.2f}".format(roc * 100),
                            Time=experiment_time)

        file_to_lock.write(scores_string)
    finally:
        unlock_file(lock_file_path)
        file_to_lock.close()


def print_scores_on_file(filepath, scores_dict):
    with open(filepath, 'w+') as f:
        f.write("Accuracy (%): {0:.2f}".format(scores_dict['Accuracy'] * 100) + "\n")
        f.write("UAR (%): {0:.2f}".format(scores_dict['UAR'] * 100) + "\n")
        cm = scores_dict['ConfMatrix']
        f.write("\n" + "Confusion Matrix" + "\n")
        f.write("\t V\t O\t T\t E" + "\n")
        f.write("V \t" + str(cm[0, 0]) + "\t" + str(cm[0, 1]) + "\t" + str(cm[0, 2]) + "\t" + str(cm[0, 3]) + "\n")
        f.write("O \t" + str(cm[1, 0]) + "\t" + str(cm[1, 1]) + "\t" + str(cm[1, 2]) + "\t" + str(cm[1, 3]) + "\n")
        f.write("T \t" + str(cm[2, 0]) + "\t" + str(cm[2, 1]) + "\t" + str(cm[2, 2]) + "\t" + str(cm[2, 3]) + "\n")
        f.write("E \t" + str(cm[3, 0]) + "\t" + str(cm[3, 1]) + "\t" + str(cm[3, 2]) + "\t" + str(cm[3, 3]) + "\n")
        f.write("\n")
        f.write("Classification Report\n" + scores_dict['Classification Report'] + "\n")


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues,
#                           filename='.'):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         cm = cm*100
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#     plt.figure()
#     plt.rcParams.update({'font.size': 26})
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     #plt.title(title)
#     plt.colorbar()
#
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#     np.set_printoptions(precision=2)
#     plt.show()
#     plt.savefig(filename)
#

def lock_file(lock_file_path):
    #logging.info("file lock attempt")
    if not os.path.isfile(lock_file_path):
        lock = open(lock_file_path, 'a+')
        lock.close()
        #logging.info("file lock success")
        return True
    else:
        #logging.info("file lock failure")
        return False


def unlock_file(lock_file_path):
    #logging.info("file unlock attempt")
    if os.path.isfile(lock_file_path):
        os.remove(lock_file_path)
        #logging.info("file unlock success")
        return True
    else:
        #logging.info("file unlock failure")
        return False


def compute_auc(data, labels, path=None):
    precision, recall, thresholds = precision_recall_curve(labels, data)
    average_precision = average_precision_score(labels, data)
    roc = roc_auc_score(labels, data)

    if path is not None:
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                       color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
        out_fig_name = os.path.join(path,'precision_recall_curve.png')
        plt.savefig(out_fig_name)

    return roc, average_precision


def compute_det(true_labels, row_predictions, th_step=None, show_figure=False, save_figure=False):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    """
    min_value = row_predictions.min()
    max_value = row_predictions.max()
    if th_step is None:
        th_step = (max_value - min_value)/100
    thresholds = np.arange(min_value, max_value, th_step)
    # thresholds = get_threshold_to_evaluate(row_predictions)

    tpr = np.zeros(len(thresholds))
    tnr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    fnr = np.zeros(len(thresholds))
    # compute prediction labels for each threshold
    for idx, t in enumerate(thresholds):
        pred_labels = threshold_row_prediction(row_predictions, t)
        tpr[idx], tnr[idx], fpr[idx], fnr[idx] = compute_TP_TN_FP_FN_rate(pred_labels, true_labels)

    axis_min = min(fpr[0], fnr[-1])
    fig, ax = plt.subplots()
    plt.plot(fpr, fnr)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Miss rate')
    plt.ylabel('False allarm rate')
    plt.title('Detection error trade-off curve')
    plt.legend(loc="lower right")
    ticks_to_use = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.axis([0.001, 1, 0.001, 1])
    if show_figure:
        fig.show()
    if save_figure:
        fig.savefig('det.png')
    del fig
    return tpr, tnr, fpr, fnr, thresholds

def compute_TP_TN_FP_FN(pred_labels, true_labels):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    return TP, TN, FP, FN


def compute_TP_TN_FP_FN_rate(pred_labels, true_labels):

    TP, TN, FP, FN = compute_TP_TN_FP_FN(pred_labels, true_labels)
    tpr = TP/(TP+FN)
    tnr = TN/(TN+FP)
    fpr = FP/(FP+TN)
    fnr = FN/(FN+TP)

    return tpr, tnr, fpr, fnr

def get_threshold_to_evaluate(row_predictions, th_step=None):

    min_value = min(row_predictions) - 1
    max_value = max(row_predictions) + 1
    if th_step is None:
        step = (max_value - min_value) / 100

    thresholds = np.arange(min_value, max_value, step)
    return thresholds

def threshold_row_prediction(row_predictions, threshold):

    pred_labels = np.zeros(len(row_predictions))
    for i, row_prediction in enumerate(row_predictions):
        if row_prediction < threshold:
            pred_labels[i] = 0
        else:
            pred_labels[i] = 1

    return pred_labels

def normalize_confusion_matrix(cm):
    '''

    :param cm: confusion matrix
    :return:
    '''
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm

def precision_recall_f1score_from_confusion_matrix(cm, index_label=None):
    """

    :param cm: confusion matrix
    :param index_label: the index of the class inside the confusion matrix
    :return:
    """
    cm = cm.astype('float')
    if index_label is not None:
        precision = cm[index_label][index_label] / cm.sum(axis=1)[index_label]
        if cm[index_label][index_label] == 0:
            recall = 0
        else:
            recall = cm[index_label][index_label] / cm.sum(axis=0)[index_label]
        if precision == 0 and recall == 0:
            f1_measure = 0
        else:
            f1_measure = 2 * precision * recall / (precision + recall)

        return precision, recall, f1_measure
    else:
        #todo compute metrics for all class and return they as a vectrors
        pass
def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure

    https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = seaborn.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def print_confusion_matrix(cm):
    string = '\n\t\t\thf\tobj\n' \
             'hf\t\t\t{0:.2f}\t{1:.2f}\n' \
             'obj\t\t\t{2:.2f}\t{3:.2f}\n'.format(cm[0][0], cm[0][1], cm[1][0], cm[1][1])
    return string