import json
import os
from glob import glob

from skimage.io import imread
from keras.metrics import BinaryIoU, BinaryAccuracy  # Accuracy
import matplotlib.pyplot as plt
import pandas as pd


def calc_metrics(y_true, y_pred, threshold: float = 0.5):
    """
    Calculates pixel-wise the BinaryAccuracy (how often a thresholded
    prediction pixel, matches the ground truth pixel), and the BinaryIoU
    (= TruePos / (TruePos + FalsePos + FalseNeg) for the foreground class
    label, i.e. 1.
    :param y_true: ground truth image or list of ground truth images [np.array]
    :param y_pred: prediction image or list of prediction images [np.array]
    :param threshold: (float) thresholding value for prediction image
    :return: tuple(list(floats)) i.e. (BinaryAccuracy-values, IoU-values)
    """

    # currently i am passing a np.array
    if not isinstance(y_true, list):
        y_true = [y_true]
    if not isinstance(y_pred, list):
        y_pred = [y_pred]

    # Accuracy, not relevant for U-Net...
    '''
    Basically compares if one value of a array matches 
    the corresponding gt value, (without thresholding the prediction).
    Hence, this is not really for U-Nets
    INFO: update_state() will average all of the added metrics
    '''
    # all_acc = []
    # acc = Accuracy()

    # Binary Accuracy
    '''
    Does the same as accuracy but on thresholded predictions.
    '''
    all_bin_acc = []
    bin_acc = BinaryAccuracy(threshold=threshold)

    # Binary IoU
    '''
    can be done for a specific target class, i.e. label.
    Uses thresholding for class(es), like BinaryAccuracy.
    iou = tp / (tp + fp + fn)
    '''
    iou = BinaryIoU(target_class_ids=[1], threshold=threshold)
    all_iou = []

    # print('bin-acc     vs     iou')
    for (_true, _pred) in zip(y_true, y_pred):
        # for accuracy (not really useful for binary results)
        # acc.reset_state()
        # acc.update_state(y_true=_true, y_pred=_pred)
        # all_acc.append(acc.result().numpy())
        # for binary accuracy
        bin_acc.reset_state()
        bin_acc.update_state(y_true=_true, y_pred=_pred)
        all_bin_acc.append(float(bin_acc.result().numpy()))
        # for IoU
        iou.reset_state()
        iou.update_state(y_true=_true, y_pred=_pred)
        all_iou.append(float(iou.result().numpy()))

    return all_bin_acc, all_iou


def load_metrics(path: str) -> dict:
    """
    Load the test metrics of the model (json-file).
    :param path: (str) to the json file.
    :return: a dictionary with metrics for (keys) model and
             best-checkpoint model
    """
    if path.endswith('.h5'):
        path = path.replace('.h5', '.json')
    if not path.endswith('.json'):
        raise IOError(f'The file is not a JSON file: <{path}>')
    # read the json file as dictionary
    f = open(path)
    data = json.load(f)
    f.close()

    # get the test_metrics, contains keys for trained model and best-ckp model

    try:
        test_metrics = data['test_metrics']
    except KeyError:
        raise KeyError(f'Could not find the "test_metrics" key in the json '
                       f'file. The <{path}> does not seem to have been '
                       f'written with this library.')
    return test_metrics


def calc_metrics_average(test_metrics: dict) -> dict:
    """
    Adds the averages over the images for the different metrics.
    Modifies/returns the input dictionary.
    :param test_metrics: (dict) from load_metrics() function, i.e.:
            - model_name.h5
               "@resolution=1/1": {
                   "binary_accuracy": {
                       - "thresholds" -> list of floats
                       - "imageXYZ.tif" -> list of floats
                       - ... }
                   "binary_iou: {...}}
            - ...
    :return: (dict) test_metrics modified
    """
    for model, resolutions in test_metrics.items():
        for res, metrics in resolutions.items():
            for metric, image_names in metrics.items():
                averages = []
                for i in range(len(image_names['thresholds'])):
                    cur_average = 0
                    for image_name, values in image_names.items():
                        if (
                            not image_name == 'thresholds' and
                            not image_names == 'averages'
                        ):
                            cur_average += values[i]
                    averages.append(cur_average / (len(image_names) - 1))
                metrics[metric]['averages'] = averages
    return test_metrics


def create_metrics_graph(test_metrics: dict, save_dir_path: str) -> dict:
    # TODO describe the return dict better
    """
    Takes the test metric metadata and plots the average metrics for the
    different resolutions and models.
    Creates a dict for that contains the best parameters to use according
    to the binary_iou metrics.
    :param test_metrics: (dict) from the calc_metrics_average() function, i.e.:
            - model_name.h5
               "@resolution=1/1": {
                   "binary_accuracy": {
                       - "thresholds" -> list of floats
                       - "imageXYZ.tif" -> list of floats
                       - ...
                       - "averages" -> list of floats}
                   "binary_iou: {...}}
            - ...
    :param save_dir_path: (str) folder path to desired saving location
    :return: (dict) per model the best parameters to use
    """
    # sanity check
    if not os.path.exists(save_dir_path):
        raise IOError(f'The chosen saving location does not exist: '
                      f'<{save_dir_path}>')
    if os.path.isfile(save_dir_path):
        # if a file was provided, take the parent folder.
        save_dir_path = os.path.dirname(save_dir_path)

    # convert dictionary to have only averages      --------------------------
    #   per resolution per metric per model
    x_axis = None
    per_model = {}
    for model, resolutions in test_metrics.items():
        per_metric = {}
        for res, metrics in resolutions.items():
            for metric, values in metrics.items():
                for name in values.keys():
                    if name == 'thresholds' and x_axis is None:
                        x_axis = values[name]
                    if name == 'averages':
                        if metric not in per_metric.keys():
                            per_metric[metric] = {}
                        if res not in per_metric[metric].keys():
                            per_metric[metric][res] = values[name]
                        else:
                            # should not be the case ever...
                            print(f'Error:found {res} already for {metric}')
        per_model[model] = per_metric

    # create and save plots         ------------------------------------------
    for model, metrics in per_model.items():
        for metric, resolutions in metrics.items():
            plt.xlabel('IoU threshold')
            plt.ylabel('Metric value')
            plt.title(model + ' - ' + metric)
            plt.axis((0.1, 0.9, 0, 1))  # set axis limits
            for res, values in resolutions.items():
                # plot only thresholds 0.1 - 0.9
                plt.plot(x_axis[1:], values[1:], label=res)
            plt.legend()
            # save plot
            path = os.path.join(save_dir_path, f'{model}-{metric}.png')
            plt.savefig(path)
            # plt.show()
            plt.clf()  # clear plot, for next one
    print(f'Metric graphs were saved to the folder: <{save_dir_path}>')

    # Find the best threshold and resolution to use for each model      ------
    best_parameters = {}
    for model, metrics in per_model.items():
        df = pd.DataFrame.from_dict(metrics['binary_iou'])
        # best resolution
        best_res = df.max().idxmax()
        # divide by 10 to get the threshold value (lucky indexing)
        best_threshold = df.idxmax()[best_res] / 10
        best_parameters[model] = {
            'best_binary_iou_parameters': {
                'best_binary_iou_value': df.max().max(),
                'best_resolution': best_res,
                'best_threshold': best_threshold
            }
        }
        # print the best metrics for model
        print(f'--> Best parameters for {model} -> threshold = '
              f'{best_threshold} {best_res} '
              f'(with a value of {df.max().max()}).')
    return best_parameters


# Testing
if __name__ == '__main__':
    json_path = 'G:/20231006_Martin/EfficientUNet/models_test/models/test_history_model/test_history_model.json'
    create_metrics_graph(
        test_metrics=load_metrics(json_path),
        save_dir_path=json_path
    )

    '''
    gt_path = 'G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks/test'
    pred_path = 'G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks/test_predictions'

    gt = glob(gt_path + '/*.tif')
    pred = glob(pred_path + '/*.tif')
    gt = [imread(x) for x in gt]
    pred = [imread(x) for x in pred]

    calc_metrics(gt, pred, threshold=0.9)
    '''
