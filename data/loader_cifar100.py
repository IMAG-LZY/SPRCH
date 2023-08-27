from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
import torch.utils.data as data


class CIFAR100_sample(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    #url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    #filename = "cifar-10-python.tar.gz"
    #tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
            self,
            root: str = 'dataset',
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            sample_data: bool = True,
            sample_num: int = 100,
            # sample_id_file: str = None,
            sample_id_file: str = './data/cifar100_sample_id_training.npy',
    ) -> None:

        #super(CIFAR10, self).__init__(root, transform=transform,
        #                              target_transform=target_transform)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.sample_num = sample_num
        self.sample_id_file = sample_id_file

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        if sample_data:
            self.sample_data_training()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def sample_data_training(self) -> None:
        if self.sample_id_file is not None:
            sample_id = np.load(self.sample_id_file)
        else:
            sample_id, sample_id_1 = self.do_sampling()
            np.save('cifar100_sample_id_training.npy', sample_id)
        sample_feat = self.data[sample_id]
        sample_targets = np.array(self.targets)[sample_id]
        self.data = sample_feat
        self.targets = list(sample_targets)

    def do_sampling(self):
        sample_id = []
        for i in range(100):
            cur_lb_id = np.array([k for (k, v) in enumerate(self.targets) if v == i])
            sample_rnd_id = np.random.choice(cur_lb_id.size, self.sample_num, replace=False)
            sample_lb_id = cur_lb_id[sample_rnd_id]
            sample_id.append(sample_lb_id)
        sample_id = np.array(sample_id).reshape(-1)
        return sample_id