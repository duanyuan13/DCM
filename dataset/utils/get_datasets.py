"""
Helper functions for get datasets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging


def get_datasets(args, dataset_name, transforms=None):
    """
    get datasets for train and validation
    Args:
        args:
            dataset:
                # root dir containing data source
                root: 'data/352_test_e300'
                # bool flag for whether or not to shuffle training set
                shuffle: True
                # number of samples for each mini-batch for training
                batch_size: 2
                # number of samples for each mini-batch for validation
                valid_batch_size: 2
                # try to use pin_memory, but if system freeze or swap being used a lot, disable it.
                pin_memory: False
                # bool flag for whether or not to drop the last batch if no enough samples left
                drop_last: False
                train:
                  name: 'Drone_train'
                  index_path: 'data/scene/train_scene_no_prefix.txt'
                val:
                  name: 'Drone_val'
                  index_path: 'data/scene/val_scene_no_prefix.txt'
                test:
                  name: 'Drone_test'
                  index_path: 'data/scene/test_scene_no_prefix.txt'
        dataset_name: args.train/val/test.name
        transforms: transformation of data
    Returns:
        dataset (torch.utils.data.Dataset)
    """
    if dataset_name == 'Drone_train':
        from ..DroneDataset import BoxedDroneDataset
        dataset = BoxedDroneDataset(root=args.train.root,
                                    index_dir=args.train.index_dir,
                                    npy_file=args.train.npy_path)
        
    elif dataset_name == 'Drone_val':
        from ..DroneDataset import BoxedDroneDataset
        dataset = BoxedDroneDataset(root=args.val.root,
                                    index_dir=args.val.index_dir,
                                    npy_file=args.val.npy_path)
    elif dataset_name == 'Drone_test':
        from ..DroneDataset import BoxedDroneDataset
        dataset = BoxedDroneDataset(root=args.test.root,
                                    index_dir=args.test.index_dir,
                                    npy_file=args.test.npy_path)
    else:
        logging.getLogger('Dataset').error(
            'Dataset for %s not implemented', dataset_name)
        raise NotImplementedError

    return dataset
