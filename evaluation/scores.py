import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report, f1_score, \
    precision_recall_curve, average_precision_score, roc_auc_score
from .dcase_framework import eval_utils
from .dcase_framework import mono_sed_eval
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt


def print_scores(scores_dict, path=None, json_name='scores.json'):
    if path:
        os.makedirs(os.path.join(path, 'scores'), exist_ok=True)
        with open(os.path.join(path, 'scores', json_name), 'w+') as outfile:
            json.dump(scores_dict, outfile, indent=4, sort_keys=True)


def compute_scores(labels, predictions, split=False):
    predictions = np.argmax(predictions, 1)
    labels = np.argmax(labels, 1)
    if split:
        predictions = np.reshape(predictions, (-1, 13))
        labels = np.reshape(labels, (-1, 13))
        preds_smoothed = list()
        labels_smoothed = list()
        for i in range(0, len(predictions)):
            preds_smoothed.append(np.argmax(np.bincount(predictions[i, :])))
            labels_smoothed.append(np.argmax(np.bincount(labels[i, :])))
        accuracy = accuracy_score(labels_smoothed, preds_smoothed)
    else:
        accuracy = accuracy_score(labels, predictions)
    return accuracy


def score_report(base_path, report_data=None):
    scores_file_path = os.path.join(base_path, 'Scores_Report.csv')
    with open(scores_file_path, "a") as score_f:
        if os.path.isfile(scores_file_path) and os.path.getsize(scores_file_path) == 0:
            # scores_string = "FIRST LINE OF RESULTS CSV"
            scores_string = " {conf_id:<25s} |  {acc_valid:<25s} | {acc_test:<25s} | {exp_time:<25s} \n".format(
                conf_id='Conf_id',
                acc_valid="Validation Accuracy",
                acc_test="Test Accuracy",
                exp_time="Exp. Time")
            score_f.write(scores_string)

        scores_string = " {conf_id:<25s} |  {acc_valid:<25s} | {acc_test:<25s} | {exp_time:<25s} \n".format(
            conf_id=report_data['conf_id'],
            acc_valid="{:4.2f}".format(report_data['valid_score']),
            acc_test="{:4.2f}".format(report_data['test_score']),
            exp_time=report_data['exp_time'])
        score_f.write(scores_string)


def sed_eval_metric(pred, y, frames_in_1_sec=50, posterior_thresh=0.5, no_bkg=True):
    scores = dict()
    if no_bkg:
        pred = pred[:, :, 1:]
        y = y[:, :, 1:]
    pred_thresh = pred > posterior_thresh # Take only event predictions
    scores['f1_overall_1sec'] = eval_utils.f1_overall_1sec(pred_thresh, y, frames_in_1_sec)
    scores['er_overall_1sec'] = eval_utils.er_overall_1sec(pred_thresh, y, frames_in_1_sec)
    return scores


def compute_auc(data, labels, path=None):
    results_dict = {'roc': 0, 'AP': 0}
    precision, recall, thresholds = precision_recall_curve(labels, data)
    results_dict['AP'] = average_precision_score(labels, data)
    results_dict['roc'] = roc_auc_score(labels, data)

    # if path is not None:
    #     plt.figure()
    #     plt.step(recall, precision, color='b', alpha=0.2, where='post')
    #     plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.ylim([0.0, 1.05])
    #     plt.xlim([0.0, 1.0])
    #     plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(results_dict['AP'] * 100))
    #     out_fig_name = os.path.join(path, 'precision_recall_curve.png')
    #     plt.savefig(out_fig_name)

    return results_dict


def threshold(data, labels, threshold=None, reshape=False, paths=None):
    # if reshape:
    #     preprocessing = preproc.DataPreprocessing(data_dims_num=2)
    #     label_dims = preprocessing.get_max_data_dimensions(labels, concatenate=True)
    #     labels = preprocessing.concatenate_data(labels, shapes=label_dims)
    #     data = preprocessing.concatenate_data(data, shapes=label_dims)

    roc_auc, average_precision = compute_auc(data, labels, path=paths)
    if threshold is not None:
        low_values_indices = data < threshold
        data[low_values_indices] = 0
        high_values_indices = data.nonzero()
        data[high_values_indices] = 1

    return data, labels, roc_auc, average_precision


def mono_sed_scores(data, labels, threshold=0.5, paths=None, verbose=0):
    thresholds = np.arange(0, threshold, 0.05, dtype=np.float32)
    er_tmp = np.inf
    best_results = {}
    for t in thresholds:
        temp_file = os.path.join(paths, 'temp_results.txt')
        result_dict = mono_sed_eval.compute_mono_sed_scores(data, labels, threshold=t, temp_file=temp_file, verbose=verbose)
        if result_dict['error_rate'] < er_tmp:
            er_tmp = result_dict['error_rate']
            result_dict['threshold'] = str(t)
            best_results = result_dict
    return best_results

