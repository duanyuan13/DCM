import os
import torch
import logging
import random
import torch.utils.data as torch_data
import numpy as np
import shutil
from tqdm import tqdm
import pandas as pd
from .drone_utils import *
from sklearn.model_selection import train_test_split
from einops import rearrange
from math import  pi


class DroneDataset(torch_data.Dataset):
    def __init__(self, root, index_dir, transform=None, seed=42):
        """
        extract file name from index_dir,
        with the imformation of root and file_name, we can gain the directory of file.
        Params:
            root: root path
            index_dir: path of txt file
            transform: transformation for data
            seed: random seed, used for split datasets based on index_dir


        For example:
        file_name = '2022-10-19-16-22-2736.txt'
        root = /home/user/datasets/
        then:
          the directory of left image: /home/user/datasets/image/2022-10-19-16-22-Left-2736.jpg
          the directory of right image: /home/user/datasets/image/2022-10-19-16-22-Right-2736.jpg
          the directory of inferenced left box: /home/user/datasets/inference/detected_labels/2022-10-19-16-22-Left-2736.txt
          the directory of inferenced right box: /home/user/datasets/inference/detected_labels/2022-10-19-16-22-Right-2736.txt

        the directory tree looks like:
        Root
            ├─image: the original images
            ├─inference: the result inferenced by object detector(YOLO, SSD, etc.).
            │  ├─boxed_images: image with bboxes.
            │  └─detected_labels: result of box, [cls, x, y, w, h, confs], which is same to YOLO formats.
            ├─label: the original label(ground truth).
            └─annotations.csv: labels with distance informantion.
        """
        self.root = root
        self.index_dir = index_dir
        self.transform = transform
        self.seed = seed
        check_if_dir_exist(self.root)
        with open(self.index_dir, 'r') as f:
            self.datasets = f.readlines()

    def __getitem__(self, index):
        file_id = self.datasets[index]
        left_box = self.get_box(file_id, 'Left')
        # left_box = np.expand_dims(left_box, axis=0)
        right_box = self.get_box(file_id, 'Right')
        # right_box = np.expand_dims(right_box, axis=0)
        distance = self.get_distance(file_id)
        if self.transform:
            left_box = self.transform(left_box)
            right_box = self.transform(right_box)
            distance = self.transform(distance)

        return left_box, right_box, distance

    def __len__(self):
        return len(self.datasets)

    def get_box(self, file_id, flags):
        boxed_txt_name = self.get_boxed_txt(file_id, flags)
        boxed_path = os.path.join(self.root, 'inference', 'detected_labels', boxed_txt_name)
        box = []
        with open(boxed_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                box.append(line.split(' '))
        return np.array(box, dtype=np.float32)

    def get_distance(self, file_id):
        boxed_txt_name = self.get_boxed_txt(file_id, 'Left')
        boxed_img_name = os.path.splitext(boxed_txt_name)[0] + '.jpg'

        distance_path = os.path.join(self.root, 'annotations.csv')

        df = pd.read_csv(distance_path, index_col=0)
        distance = df.loc[boxed_img_name, 'distance']

        return np.array(distance, dtype=np.float32)

    def get_boxed_txt(self, file_id: str, flags: str) -> str:
        """
        file_id: looks like this: 2022-10-19-16-22-2736.txt
        flags: can only be "Left" or "Right"
        return:
            txt_name, the name of txt file which contains box information,
            based on flags, eg: 2022-10-19-16-22-flags-2736.txt
        """
        # remove '\n'
        file_id = file_id.strip('\n')
        split_id = file_id.split('-')

        if not isinstance(file_id, str) or not isinstance(flags, str):
            raise ValueError(f'{file_id} and {flags} are not string input.')
        if (flags == 'Left') or (flags == 'Right'):
            txt_name = split_id[:-1] + [flags] + [split_id[-1]]
            return '-'.join(txt_name)
        else:
            raise ValueError(f"flags {flags} requires either 'Left' or 'Right'.")

    @staticmethod
    def check_labels(txt_dir, output_path='output.txt'):
        """
        Generate a txt file with index,
        if indexed txt(train.txt, test.txt, etc.) file not provided.
        txt_dir: directory contains boxed txt files. eg: './detected_labels/'
        output_path: path of output, eg: 'output.txt'

        output looks like: (Remove the Left and Right string)
        2022-10-19-16-22-2736.txt
        """
        check_if_dir_exist(txt_dir, is_raise_error=True)
        output = []
        for root, dirname, filenames in os.walk(txt_dir):
            for filename in filenames:
                if os.path.splitext(filename)[-1] == '.txt':
                    split_name = filename.split('-')

                    if 'Left' in split_name:
                        respect_file_name = split_name
                        respect_file_name[-2] = 'Right'
                    elif 'Right' in split_name:
                        respect_file_name = split_name
                        respect_file_name[-2] = 'Left'
                    else:
                        raise ValueError(f'file {filename} does not have either Left or Right images')
                    respect_file_name = '-'.join(respect_file_name)

                    if respect_file_name in filenames:
                        index_file = split_name[:-2]
                        index_file.append(split_name[-1])
                        index_file = '-'.join(index_file)
                        if index_file in output:
                            pass
                        else:
                            output.append(index_file)
                            with open(output_path, 'a') as f:
                                f.writelines(index_file + '\n')

    @staticmethod
    def clean_labels(dir):
        """
        some txt file contains '_', rather than '-', modified all name to
        """
        for root, dirname, filenames in os.walk(dir):
            for filename in filenames:
                    # clean the data with '-'(all str should be connected by '-', rather than '_')
                split_name = filename.split('_')
                if len(split_name) == 1:
                    pass
                else:
                    split_name = '-'.join(split_name)
                    rename_path = os.path.join(root, split_name)
                    source_path = os.path.join(root, filename)
                    os.rename(source_path, rename_path)
                    print(f'{filename} has been modified into {split_name}')

    def split_train_test_valid(self, train_ratio=0.6, test_ratio=0.2, no_prefix=True, is_abs_dir=True):
        """
        return: the output txt file contains paths of datasets
        if no_prefix = True,
            return the file without any prefix, for example:2022-10-19-16-22-2736.txt
        if is_abs_dir = True,
            return the file with absolute path, for example:
                /home/user/datasets/image/2022-10-19-16-22-Right-2736.jpg
                /home/user/datasets/image/2022-10-19-16-22-Left-2736.jpg
        """
        assert train_ratio + test_ratio <= 1, f"train_ratio:{train_ratio} + test_ratio:{test_ratio} is bigger than 1"
        valid_ratio = 1 - train_ratio - test_ratio
        random.seed(self.seed)
        train_num, test_num, valid_num = (int(train_ratio*len(self.datasets)),
                                          int(test_ratio*len(self.datasets)),
                                          int(valid_ratio*len(self.datasets)))
        train_val_list, test_list = train_test_split(self.datasets, train_size=train_num+valid_num,
                                                     test_size=test_num, random_state=self.seed)
        train_list, val_list = train_test_split(train_val_list, train_size=train_num,
                                                test_size=valid_num, random_state=self.seed)
        if no_prefix:
            no_prefix_train_dir = 'train_no_prefix.txt'
            no_prefix_test_dir = 'test_no_prefix.txt'
            no_prefix_valid_dir = 'val_no_prefix.txt'
            self.write_list_lines(no_prefix_train_dir, train_list)
            self.write_list_lines(no_prefix_test_dir, test_list)
            self.write_list_lines(no_prefix_valid_dir, val_list)
        if is_abs_dir:
            abs_prefix = os.path.join(self.root, 'image')
            abs_train_dir= 'train.txt'
            abs_test_dir = 'test.txt'
            abs_valid_dir = 'val.txt'
            self.write_abs_lines(abs_train_dir, train_list, abs_prefix)
            self.write_abs_lines(abs_test_dir, test_list, abs_prefix)
            self.write_abs_lines(abs_valid_dir, val_list, abs_prefix)
        print('done')


    def write_abs_lines(self, filedir, itemlist, abs_prefix):
        for item in itemlist:
            left_file_name = self.get_boxed_txt(item, 'Left')

            left_file_name = left_file_name.split('.')[0] + '.jpg'

            right_file_name = self.get_boxed_txt(item, 'Right')

            right_file_name = right_file_name.split('.')[0] + '.jpg'

            abs_prefix_left_path = os.path.join(abs_prefix, left_file_name)

            abs_prefix_right_path = os.path.join(abs_prefix, right_file_name)
            with open(filedir, 'a') as f:
                f.writelines(abs_prefix_left_path + '\n')
                f.writelines(abs_prefix_right_path + '\n')

    @staticmethod
    def write_list_lines(filedir, itemlist):
        with open(filedir, "w") as f:
            for item in itemlist:
                f.write(item)

    def prefix_to_abs(self, target_dir):
        """
        transpose dir without prefix into absolute dir.
        For example:
        the  dir without prefix contains:  2022-10-19-16-22-2736.txt
        the root is: /home/user/
        the target_dir is: train.txt
        then, the file train.txt contains: /home/user/images/2022-10-19-16-22-Left-2736.jpg
                                           /home/user/images/2022-10-19-16-22-Right-2736.jpg
        """
        abs_prefix = os.path.join(self.root, 'image')
        self.write_abs_lines(target_dir, self.datasets, abs_prefix)


class BoxedDroneDataset(torch_data.Dataset):
    def __init__(self, root, index_dir, mode='top_shape', npy_file=None, npy_save_path=None, transform=None):
        """
        This class is based on trained detector models, first extract the results of detector models,
        then make the result to become dataset of our disparity predict model and grain-fined model.

        extract file name from index_dir,
        with the imformation of root and file_name, we can gain the directory of file.
        Params:
            root: root path
            index_dir: path of indexed txt file
            transform: transformation for data
            mode: if the mode = 'one', the dataset only return one bbox at a time
                  else the model = 'many', the dataset return all bbox at a time

        For example:
        file_name = '2022-10-19-16-22-2736.txt'
        root = /home/user/datasets/
        then:
          the directory of left image: /home/user/datasets/image/2022-10-19-16-22-Left-2736.jpg
          the directory of right image: /home/user/datasets/image/2022-10-19-16-22-Right-2736.jpg
          the directory of inferenced left box: /home/user/datasets/inference/detected_labels/2022-10-19-16-22-Left-2736.txt
          the directory of inferenced right box: /home/user/datasets/inference/detected_labels/2022-10-19-16-22-Right-2736.txt

        the directory tree looks like:
        Root
            ├─image: the original images
            ├─inference: the result inferenced by object detector(YOLO, SSD, etc.).
            │  ├─boxed_images: image with bboxes.
            │  └─detected_labels: result of box, [cls, x, y, w, h, confs], which is same to YOLO formats.
            ├─label: the original label(ground truth).
            └─annotations.csv: labels with distance informantion.
        """
        self.root = root
        self.index_dir = index_dir
        self.transform = transform
        check_if_dir_exist(self.root)
        with open(self.index_dir, 'r') as f:
            self.datasets = f.readlines()
        if npy_file is not None:
            self.npy_dataset = np.load(npy_file, allow_pickle=True)
            self.npy_dataset = self.npy_dataset.tolist()
        else:
            self.npy_dataset = self.generate_npy_dataset(npy_save_path)

        if mode == 'top_shape' or mode == 'many':
            self.mode = mode
        else:
            logging.getLogger('Dataset').error(
                'Dataset for %s not implemented', mode)
            raise NotImplementedError

    @property
    def read_npy_dataset(file):
        npy_dataset = np.load(file)
        npy_dataset = npy_dataset.tolist()
        return npy_dataset

    def generate_npy_dataset(self, save_path=None):
        """
        store all results into one file, and send it to RAM.
        (This function is made for accelarating training speed.)
        """
        print('generate npy formats dataset...')
        npy_dataset = []
        for index, data in tqdm(enumerate(self.datasets)):
            left_box, right_box, distance, file_id = self.__getitem(index)
            dict_npy = {'left': left_box, 'right': right_box, 'dist': distance, 'file_id': file_id}
            npy_dataset.append(dict_npy)
        if save_path is not None:
            np.save(save_path, npy_dataset)
        return npy_dataset

    def __getitem(self, index):
        file_id = self.datasets[index]
        left_box = self.get_box(file_id, 'Left')
        # left_box = np.expand_dims(left_box, axis=0)
        right_box = self.get_box(file_id, 'Right')
        # right_box = np.expand_dims(right_box, axis=0)
        distance = self.get_distance(file_id)
        return left_box, right_box, distance, file_id

    def __getitem__(self, index):
        data = self.npy_dataset[index]
        file_id = data['file_id']
        distance = data['dist']
        if self.mode == 'top_shape':
            # return [x y w h] with the highest confidence among the bboxes.
            left_shape = data['left'][-1, 1:5]
            right_shape = data['right'][-1, 1:5]
            input = np.row_stack((left_shape, right_shape))
            if self.transform:
                # input shape: [2, 4]
                input = torch.tensor(input)
                distance = torch.tensor(distance)
            return input, distance, file_id
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.npy_dataset)

    def get_box(self, file_id, flags):
        boxed_txt_name = self.get_boxed_txt(file_id, flags)
        boxed_path = os.path.join(self.root, 'inference', 'detected_labels', boxed_txt_name)
        box = []
        with open(boxed_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                box.append(line.split(' '))
        return np.array(box, dtype=np.float32)

    def get_distance(self, file_id):
        boxed_txt_name = self.get_boxed_txt(file_id, 'Left')
        boxed_img_name = os.path.splitext(boxed_txt_name)[0] + '.jpg'

        distance_path = os.path.join(self.root, 'annotations.csv')

        df = pd.read_csv(distance_path, index_col=0)
        distance = df.loc[boxed_img_name, 'distance']

        return np.array(distance, dtype=np.float32)

    @staticmethod
    def get_boxed_txt(file_id: str, flags: str) -> str:
        """
        file_id: looks like this: 2022-10-19-16-22-2736.txt
        flags: can only be "Left" or "Right"
        return:
            txt_name, the name of txt file which contains box information,
            based on flags, eg: 2022-10-19-16-22-flags-2736.txt
        """
        # remove '\n'
        file_id = file_id.strip('\n')
        split_id = file_id.split('-')

        if not isinstance(file_id, str) or not isinstance(flags, str):
            raise ValueError(f'{file_id} and {flags} are not string input.')
        if (flags == 'Left') or (flags == 'Right'):
            txt_name = split_id[:-1] + [flags] + [split_id[-1]]
            return '-'.join(txt_name)
        else:
            raise ValueError(f"flags {flags} requires either 'Left' or 'Right'.")

    @staticmethod
    def check_labels(txt_dir, output_path='output.txt'):
        """
        Generate a txt file with index,
        if indexed txt(train.txt, test.txt, etc.) file not provided.
        txt_dir: directory contains boxed txt files. eg: './detected_labels/'
        output_path: path of output, eg: 'output.txt'

        output looks like: (Remove the Left and Right string)
        2022-10-19-16-22-2736.txt
        """
        check_if_dir_exist(txt_dir, is_raise_error=True)
        output = []
        for root, dirname, filenames in os.walk(txt_dir):
            for filename in filenames:
                if os.path.splitext(filename)[-1] == '.txt':
                    split_name = filename.split('-')
                    if 'Left' in split_name:
                        respect_file_name = split_name
                        respect_file_name[-2] = 'Right'
                    elif 'Right' in split_name:
                        respect_file_name = split_name
                        respect_file_name[-2] = 'Left'
                    else:
                        raise ValueError(f'file {filename} does not have either Left or Right images')
                    respect_file_name = '-'.join(respect_file_name)
                    
                    if respect_file_name in filenames:
                        index_file = split_name[:-2]
                        index_file.append(split_name[-1])
                        index_file = '-'.join(index_file)
                        if index_file in output:
                            pass
                        else:
                            output.append(index_file)
                            with open(output_path, 'a') as f:
                                f.writelines(index_file + '\n')

    @staticmethod
    def clean_labels(dir):
        """
        some txt file contains '_', rather than '-', modified all name to '-'
        """
        for root, dirname, filenames in os.walk(dir):
            for filename in filenames:
                    # clean the data with '-'(all str should be connected by '-', rather than '_')
                split_name = filename.split('_')
                if len(split_name) == 1:
                    pass
                else:
                    split_name = '-'.join(split_name)
                    rename_path = os.path.join(root, split_name)
                    source_path = os.path.join(root, filename)
                    os.rename(source_path, rename_path)
                    print(f'{filename} has been modified into {split_name}')

    def write_abs_lines(self, filedir, itemlist, abs_prefix):
        for item in itemlist:
            left_file_name = self.get_boxed_txt(item, 'Left')
            # change the suffix to jpg
            left_file_name = left_file_name.split('.')[0] + '.jpg'

            right_file_name = self.get_boxed_txt(item, 'Right')
            right_file_name = right_file_name.split('.')[0] + '.jpg'

            abs_prefix_left_path = os.path.join(abs_prefix, left_file_name)

            abs_prefix_right_path = os.path.join(abs_prefix, right_file_name)
            with open(filedir, 'a') as f:
                f.writelines(abs_prefix_left_path + '\n')
                f.writelines(abs_prefix_right_path + '\n')

    @staticmethod
    def write_list_lines(filedir, itemlist):
        with open(filedir, "w") as f:
            for item in itemlist:
                f.write(item)

    def prefix_to_abs(self, target_dir):
        """
        transpose dir without prefix into absolute dir.
        For example:
        the  dir without prefix contains:  2022-10-19-16-22-2736.txt
        the root is: /home/user/
        the target_dir is: train.txt
        then, the file train.txt contains: /home/user/images/2022-10-19-16-22-Left-2736.jpg
                                           /home/user/images/2022-10-19-16-22-Right-2736.jpg
        """
        abs_prefix = os.path.join(self.root, 'image')
        self.write_abs_lines(target_dir, self.datasets, abs_prefix)

    @staticmethod
    def cal_rotate_dist(inp, image_shape=(1280, 640), is_norm=True):
        """
        REFS: https://stackoverflow.com/questions/68800793/calculate-angles-in-an-image-python
        Args:
            inp: [B, 2, 4], B is the batch size of inp
            image_shape:
            is_norm:
        """
        w, h = image_shape
        inp_clone = inp.clone().detach() # avoid changing inp
        inp_clone = rearrange(inp_clone, 'b m n -> (b m) n')  # [2*B, 4]

        inp_xy = inp_clone[:, :2]  # [2*B, 2]
        inp_xy[:, 0] = inp_xy[:, 0]
        inp_xy[:, 1] = inp_xy[:, 1]

        inp_vec = inp_xy
        inp_vec[:, 0] = w * inp_xy[:, 0] - w / 2
        inp_vec[:, 1] = h * inp_xy[:, 1] - h / 2

        inp_norm = torch.linalg.norm(inp_vec[:, ], axis=1, keepdims=True)
        inp_vec = inp_vec[:, ] / inp_norm

        inp_rad = torch.atan2(inp_vec[:, 0], inp_vec[:, 1])
        inp_rad = (pi - inp_rad) % (2 * pi)

        # angles = torch.rad2deg(inp_rad)
        # print(angles)

        if is_norm:
            inp_norm /= torch.sqrt(torch.tensor((h/2)*(h/2)+(w/2)*(w/2)))
            # angles /= 360
            inp_clone[:, 0] = inp_rad / (2*pi)
            inp_clone[:, 1] = inp_norm.reshape(inp_norm.shape[0])

        return inp_clone


def move_images(txt_file, target_dir):
    check_if_dir_exist(target_dir)
    with open(txt_file, 'r') as f:
        path_lines = f.readlines()
        for path in tqdm(path_lines):
            path = path.split('\n')[0]
            shutil.copy2(path, target_dir)
    print('done')









