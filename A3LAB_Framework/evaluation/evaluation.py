from A3LAB_Framework.preprocessing import preprocessing as preproc
from A3LAB_Framework.reporting.scores import compute_auc


def threshold(data, labels, threshold=None, reshape=False, paths=None):
    if reshape:
        preprocessing = preproc.DataPreprocessing(data_dims_num=2)
        label_dims = preprocessing.get_max_data_dimensions(labels, concatenate=True)
        labels = preprocessing.concatenate_data(labels, shapes=label_dims)
        data = preprocessing.concatenate_data(data, shapes=label_dims)

    roc_auc, average_precision = compute_auc(data, labels, path=paths)
    if threshold is not None:
        low_values_indices = data < threshold
        data[low_values_indices] = 0
        high_values_indices = data.nonzero()
        data[high_values_indices] = 1

    return data, labels, roc_auc, average_precision
