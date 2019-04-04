import numpy as np
import re
from pathlib import PosixPath

from PIL import Image
from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset

from m2det.datasets.toplogo10s.tl10labels import TL10Labels
from m2det.errors import Errors


class BaseTL10ChildDataSet(Dataset, metaclass=ABCMeta):

    IMG_SHAPE = (256, 256)
    IMG_DIM = 3

    def __init__(self, dataset_manager):
        """

        :param dataset_manager: (m2det.datasets.toplogo10s.toplogo10s.TopLogo10s)
        """

        self.root_path = dataset_manager.root_path
        self.transforms = dataset_manager.transforms

        ### path
        self.img_root = self.root_path/"jpg"
        self.data_name_root = self.root_path/"ImageSets"
        self.bbox_root = self.root_path/"masks"

    @property
    @abstractmethod
    def name_list(self):
        pass

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
            image: (torch.Tensor)
                shape -> (N_dim, height, width)
        """
        name = self.name_list[idx]

        img_path = self.__get_path_from_(name=name)
        img = self.__get_img_from_(img_path)

        target = self.__get_target_from_(name=name)
        #TODO: tansforms -> preproc
        img, target = self.transforms(img, target)

        return img, target

    def __get_path_from_(self, name):
        label = self.__get_label_from_(name=name)
        path = self.img_root/label/"{}.jpg".format(name)
        return path

    def __get_img_from_(self, path: PosixPath):
        if not path.exists():
            raise Errors().FileNotFound(path=path)

        # img = Image.open(str(path))
        import cv2
        img = cv2.imread(str(path))
        # if self.transforms:
        #     img = self.transforms(img)
        return img

    def __get_label_from_(self, name):
        """
        To remove number of end of the image name
        :param name:
            a name from text in the ImageSets directory.
        :return:
        """
        label = str(re.sub("[0-9]+$", "", name)).lower()
        if label == "adidas":
            label = "adidas0"

        return label

    def __get_target_from_(self, name):
        """
        target has coordinates and the label
        :param idx:
        :param label:
        :return:
        """
        label = self.__get_label_from_(name=name)
        int_label = int(getattr(TL10Labels, label))
        bbox_coords_path = self.bbox_root/label/"{}.jpg.bboxes.txt".format(name)

        if not bbox_coords_path.exists():
            raise Errors().FileNotFound(path=bbox_coords_path)
        with bbox_coords_path.open() as f:
            bbox_txt = f.read()

        if not bbox_txt:
            msg = Errors.BASE_MSG
            msg += "NOT correct values\n"
            msg += "\tpath: {path}".format(path=bbox_coords_path)
            raise ValueError(msg)

        targets = []
        for one_bbox in bbox_txt.split("\n"):
            if not one_bbox:
                continue
            xmin, ymin, xmax, ymax = one_bbox.split(" ")
            targets.append([xmin, ymin, xmax, ymax, int_label])

        if len(targets) == 0:
            msg = Errors.BASE_MSG
            msg += "NOT correct values\n"
            msg += "\tpath: {path}".format(path=bbox_coords_path)
            raise ValueError(msg)

        return np.array(targets, dtype=np.float64)


