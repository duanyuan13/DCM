import numpy as np
import torch
import os
import ntpath

def preds_filter(prediction, conf_thres=0.2, num_points_proposal=10):
    """Runs filtered predictions on inference results

    Returns:
         tensor: [batch, x1, y1, x2, y2, confs, cls], where xy1=top-left, xy2=bottom-right
    """

    filtered_confs_preds = prediction[..., 4] > conf_thres  # candidates

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for index_image, preds in enumerate(prediction):  # image index, image inference

        preds = preds[filtered_confs_preds[index_image]]  # confidence

        # If none remain process next image
        if not preds.shape[0]:
            continue
        np_preds = preds.cpu().numpy()
        # sort predictions based on confs, from large to small
        sorted_preds = np_preds[np.lexsort(-np_preds[:, :5].T)]
        # check if points in the box which has highest confs.
        sorted_preds = check_is_xy_in_box(sorted_preds).reshape(-1, 6)
        # If results less than num_points_proposal, pad with 0
        # the pad way can choose by set params of np.pad
        if len(sorted_preds) < num_points_proposal:
            pad_tensor = np.zeros(((num_points_proposal-len(sorted_preds)), sorted_preds.shape[1]))
            sorted_preds = np.row_stack((sorted_preds, pad_tensor))
            # sorted_preds = np.pad(sorted_preds, ((0, num_points_proposal-len(sorted_preds)), (0, 0)), 'constant')

        # convert xywh2xyxy
        box_xyxy = xywh2xyxy(sorted_preds[:, :4])
        sorted_preds = np.column_stack((box_xyxy, sorted_preds[:,4:]))
        output[index_image] = torch.tensor(sorted_preds[:num_points_proposal, :], device=prediction.device)

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def check_is_xy_in_box(sorted_preds):
    anchor_x, anchor_y, anchor_w, anchor_h= sorted_preds[0][:4]
    boxed_predictions = sorted_preds[0]
    for pred in sorted_preds[1:]:
        coord_x, coord_y = pred[0], pred[1]
        if abs(coord_x-anchor_x) < anchor_w and abs(coord_y - anchor_y) < anchor_h:
            boxed_predictions = np.row_stack((boxed_predictions, pred))

    return boxed_predictions


def extract_filename(filepath: str) -> str:
    """
    Ref: https://www.alpharithms.com/how-to-get-a-filename-from-a-path-in-python-431623/
    Given a filepath string, extract the filename regardless of file separator type
    or presence in the trailing position.
    Args:
        filepath: the filepath string from which to extract the filename
    Returns:
        str object representing the file
    For example:
    input:"C:/path/to/my/file.ext"
    return: "file.ext"
    """
    head, tail = ntpath.split(filepath)
    return tail or ntpath.basename(head)


def check_if_dir_exist(path, is_raise_error=False):
    if is_raise_error:
        if not os.path.isdir(path):
            raise ValueError(f'dirtory: {path} is not exist.')
    if not os.path.isdir(path):
        os.mkdir(path)
        print("directory is not exist, make the directory..")
    else:
        print(path, "has been existed.")


def modified_label(source_dir, target_label, output_dir='output'):
    """
    This code is designed for YOLO labels.
    Given a labeled data,
    this code changes the class number to the target label.

    for example:
    source_dir contains the information:
    15 0.560937 0.485417 0.189062 0.212500
    then the modified information is:
    0 0.560937 0.485417 0.189062 0.212500

    pramas:
    source_dir: the directory stored labeled files
    target_labe: label to change
    output_dir: the directory of output
    """
    check_if_dir_exist(output_dir)
    for root, dirname, filenames in os.walk(source_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] == '.txt':
                txt_dir = os.path.join(root, filename)
                target_dir = os.path.join(output_dir, filename)
                with open(txt_dir, "r") as f, open(target_dir, 'a') as ff:
                    data = f.readline()
                    split_data = data.split()
                    split_data[0] = str(target_label)
                    target_data = ' '.join(split_data) + '\n'
                    ff.writelines(target_data)


def modified_txt(source_dir, target_dir, output_dir='output'):
    """
    This code is designed for YOLO labels.

    Given a file contains the directory of datasets,
    this code change the path into a new directory.

    This code usually works for datasets that it's path has been changed.

    """
    base_source_file_name = extract_filename(source_dir)
    check_if_dir_exist(output_dir)
    output_dir = os.path.join(output_dir, base_source_file_name)
    with open(source_dir, 'r') as f, open(output_dir, 'a') as ff:
        for line in f:
            base_file_name = extract_filename(line)
            target_file_dir = os.path.join(target_dir, base_file_name)
            ff.writelines(target_file_dir)
    print('Complete')