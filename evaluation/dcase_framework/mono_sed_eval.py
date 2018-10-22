from copy import deepcopy
import numpy as np
import sed_eval
import logging
from scipy import signal
from A3LAB_Framework.utility.utility import item_to_dictionary


def onset_detection(preds, threshold=0.5, chunk_size=20, shift=0, win_length=50, median_filt_kernel=25):

    M = win_length
    tau = -(M - 1) / np.log(0.01)
    window = signal.exponential(M, 0, tau, False)

    onsets = []
    y_pred_proc = []

    onset = []
    for seq in preds:
        onset.append(seq[:, 1])
    preds = np.asarray(onset)
    del onset

    for seq in preds:
        # seq = signal.convolve(seq, window, mode='same', method='auto')
        seq = signal.medfilt(seq, kernel_size=median_filt_kernel)
        y_pred_proc.append(deepcopy(seq))
    # normalization
    y_pred_proc = np.asarray(y_pred_proc, dtype=np.float32)
    # y_pred_proc /= y_pred_proc.max()

    # Thresholding
    i = 0
    for seq in deepcopy(y_pred_proc):
        low_values_indices = seq < threshold
        seq[low_values_indices] = 0
        high_values_indices = seq.nonzero()
        seq[high_values_indices] = 1
        if chunk_size is not None:
            idxs = find_contiguous_regions(seq, chunk=chunk_size, shift=shift)
            out_seq = np.zeros_like(seq, dtype=np.int8)
            for elem in idxs:
                out_seq[elem[0]:elem[1]] = 1
                counter = np.nonzero(out_seq)[0].size
                rest = counter % int(chunk_size)
                if rest != 0:
                    if elem[1] + chunk_size - rest < len(out_seq):
                        out_seq[elem[0]:elem[1] + (chunk_size - rest)] = 1
            onsets.append(out_seq)
        else:
            onsets.append(seq)
        i = i + 1

    return onsets


def find_contiguous_regions(activity_array, hop_size=None, shift=0, chunk=None):
    """Find contiguous regions from bool valued numpy.array.
    Transforms boolean values for each frame into pairs of onsets and offsets.
    Parameters
    ----------
    activity_array : numpy.array [shape=(t)]
        Event activity array, bool values
    hop_size: time resolution in sec
    Returns
    -------
    change_indices : numpy.ndarray [shape=(2, number of found changes)]
        Onset and offset indices pairs in matrix
    """

    # Find the changes in the activity_array
    change_indices = np.diff(activity_array).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    if shift != 0 and len(change_indices) != 0:
        # check se lo shift manda fuori dalla lunghezza massima
        last = change_indices[-1]
        if last + shift < len(activity_array):
            change_indices = change_indices + shift
        else:
            diff = last + shift - (len(activity_array) - 1)
            change_indices = change_indices + shift
            change_indices[-2:] = change_indices[-2:] - diff

    # Reshape the result into two columns
    if len(change_indices) % 2 != 0:
        change_indices = change_indices[:-1]
    change_indices = change_indices.reshape((-1, 2))
    elem_tmp = np.zeros(2, dtype=int)
    if chunk is not None:
        for elements in change_indices:
            length = elements[1] - elements[0]
            rest = length % chunk
            if rest != 0:
                if rest > (chunk / 2):
                    if elements[1] + (chunk - rest) < len(activity_array):
                        elements[1] = elements[1] + (chunk - rest)
                    else:
                        elements[0] = elements[0] - (chunk - rest)
                else:
                    elements[1] = elements[1] - rest
                if elem_tmp[1] > elements[0]:
                    if elements[0] != elements[1]:
                        length_2 = elem_tmp[1] - elements[0]
                        rest_2 = length_2 % chunk
                        if elements[1] + rest_2 < len(activity_array):
                            elements[1] = elements[1] + rest_2
                        else:
                            elements[1] = elements[0]
            elem_tmp = deepcopy(elements)

    if hop_size is not None:
        change_indices = change_indices * hop_size

    return change_indices


def compute_mono_sed_scores(preds, labels_dataframe, threshold=0.5, onset_arch=True, temp_file='results.txt', verbose=0):
    onsets = onset_detection(preds, threshold=threshold)
    events = []
    for binary_seq in onsets:
        if binary_seq.ndim > 1:
            binary_seq = binary_seq[:, 0]
        events.append(find_contiguous_regions(binary_seq, hop_size=0.02, shift=0))
    event_based_metric, segment_based_metric = sed_eval_score(labels_dataframe,
                                                              onset_arch=onset_arch,
                                                              event_list=events,
                                                              temp_res_filename=temp_file,
                                                              verbose=verbose)
    results = {'error_rate': event_based_metric['overall']['error_rate']['error_rate'],
               'f1_score': event_based_metric['overall']['f_measure']['f_measure']}
    return results


def sed_eval_score(dataset_metadata=None,
                   onset_arch=True,
                   class_labels=['background', 'babycry', 'glassbreak', 'gunshot'],
                   event_list=[],
                   temp_res_filename='results.txt',
                   verbose=0):
    res_f = open(temp_res_filename, 'w')
    for i in range(0, len(event_list)):
        predictions = event_list[i]
        file_id = dataset_metadata.ix[i]['filename']
        if 'babycry' in file_id:
            event_name = 'babycry'
        elif 'glassbreak' in file_id:
            event_name = 'glassbreak'
        elif 'gunshot' in file_id:
            event_name = 'gunshot'
        if onset_arch:
            for c in range(0, len(predictions)):
                segment = predictions[c]
                seg_to_string = '\t'.join(["%.3f" % number for number in segment])
                #string = file_id + '\t' + seg_to_string + '\t' + 'event_onset' + '\n'
                string = file_id + '\t' + seg_to_string + '\t' + event_name + '\n'
                res_f.write(string)
        else:
            for c in range(1, len(predictions)):
                for segment in predictions[c]:
                    seg_to_string = '\t'.join(["%.3f" % number for number in segment])
                    string = file_id + '\t' + seg_to_string + '\t' + class_labels[c] + '\n'
                    res_f.write(string)
    res_f.close()

    # Create metrics classes, define parameters
    # if onset_arch:
    #     event_labels = ['event_onset']
    # else:
    #     event_labels = class_labels[1:]
    event_labels = class_labels[1:]

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=event_labels,
        time_resolution=1.0,
    )

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=event_labels,
        evaluate_onset=True,
        evaluate_offset=False,
        t_collar=0.5,
        percentage_of_length=0.5
    )

    system_decisions_list = sed_eval.io.load_event_list(temp_res_filename)
    file_idx = 0
    line = 0
    for file_id in dataset_metadata['filename'].tolist():
        meta_list = []
        meta = item_to_dictionary(dataset_metadata.ix[file_idx],
                                  keys=['filename', 'event_onset', 'event_offset', 'event_label'])
        file_idx += 1
        if meta['event_label'] == 'babycry' or meta['event_label'] == 'glassbreak' or meta['event_label'] == 'gunshot':
            # if onset_arch:
            #     meta['event_label'] = 'event_onset'
            meta_list.append(meta)

        current_file_results = []
        while line < len(system_decisions_list) and system_decisions_list[line]['filename'] == file_id:
            if 'event_label' in system_decisions_list[line]:
                if system_decisions_list[line]['event_label'] == 'babycry' \
                        or system_decisions_list[line]['event_label'] == 'glassbreak' \
                        or system_decisions_list[line]['event_label'] == 'gunshot':
                        # or system_decisions_list[line]['event_label'] == 'event_onset':
                    current_file_results.append(system_decisions_list[line])
            line += 1

        event_based_metric.evaluate(
            reference_event_list=meta_list,
            estimated_event_list=current_file_results
        )

        segment_based_metric.evaluate(
            reference_event_list=meta_list,
            estimated_event_list=current_file_results
        )

    if verbose:
        logging.info(event_based_metric)

    return event_based_metric.results(), segment_based_metric.results()
