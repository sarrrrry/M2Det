from pathlib import PosixPath, Path

import numpy as np
import torch
from torch.utils.data import Dataset

from enum import IntEnum
class WebLogoLabels(IntEnum):
    adidas = 1



class WebLogoDataset(Dataset):
    IMG_SHAPE = (256, 256)
    IMG_DIM = 3

    def __init__(self, root_path: PosixPath or str):
        """

        :param root_path:
            required directory
            hogehoge

        """
        if root_path:
            root_path = Path(root_path)
            if not root_path.exists():
                msg = str(
                    "Error: NOT Exists: {}".format(root_path)
                )

        self.img_paths = [path for path in root_path.glob("*.png")]
        weblogo = WebLogoLabels.adidas
        print(weblogo)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
            image: (torch.Tensor)
                shape -> (N_dim, height, width)
        """
        #TODO: 画像を取ってくる使用に変更する

        img = torch.zeros((3, 256, 256))
        target = np.zeros((2, 5))
        return img, target




if __name__ == '__main__':

    weblogo = WebLogoDataset(root_path="")
