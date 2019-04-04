import unittest

import numpy as np
import torch
from abc import ABCMeta, abstractmethod

from m2det.datasets.toplogo10s.toplogo10s import TopLogo10s
from m2det.datasets.weblogo import WebLogoDataset


from torchvision import transforms
class TestWebLogoDataset(unittest.TestCase):
    def setUp(self):
        root_path = ""
        self.dataset = WebLogoDataset(root_path=root_path)

    @unittest.skip("あとで")
    def test_ds_object_has_returned_img_shape(self):
        ret_img, _ = self.dataset.__getitem__(0)

        logo_dim = self.dataset.IMG_DIM
        logo_shape = self.dataset.IMG_SHAPE

        expect_shape = (logo_dim, *logo_shape)

        self.assertEqual(expect_shape, ret_img.shape)

    @unittest.skip("あとで")
    def test_weblogo_returns_img_is_tensor(self):

        ret_img, _ = self.dataset.__getitem__(0)

        self.assertTrue(isinstance(ret_img, torch.Tensor))

    @unittest.skip("あとで")
    def test_weblogo_returns_target_is_numpyndarray(self):

        _, target = self.dataset.__getitem__(0)

        self.assertTrue(isinstance(target, np.ndarray))

    @unittest.skip("あとで")
    def test_weblogo_returns_target_is_shape_of_Nx5(self):
        _, target = self.dataset.__getitem__(0)
        _, coords = target.shape
        self.assertEqual(coords, 5)



class TestTopLogo10Train(unittest.TestCase):

    @property
    def dataset(self):
        """
        e.g.)
            tf = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            root_path = "/raid/projects/logo_detection/M2Det/datasets/toplogo10"
            dataset = TopLogo10s(root_path=root_path,transforms=tf).train
            return dataset
        """

        tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        root_path = "/raid/projects/logo_detection/M2Det/datasets/toplogo10"
        return TopLogo10s(root_path=root_path,transforms=tf).train

    def test_ds_object_has_returned_img_shape(self):
        ret_img, _ = self.dataset.__getitem__(0)

        logo_dim = self.dataset.IMG_DIM
        logo_shape = self.dataset.IMG_SHAPE

        expect_shape = (logo_dim, *logo_shape)

        self.assertEqual(expect_shape, ret_img.shape)

    def test_TopLogo10_returns_tensorized_img(self):
        ret_img, _ = self.dataset.__getitem__(0)
        self.assertTrue(isinstance(ret_img, torch.Tensor))

    def test_TopLogo10_returns_npndarrayized_target(self):
        _, target = self.dataset.__getitem__(0)
        self.assertTrue(isinstance(target, np.ndarray))

    def test_TopLogo10_returns_the_target_which_is_shape_of_Nx5(self):
        _, target = self.dataset.__getitem__(0)
        _, coords = target.shape
        self.assertEqual(coords, 5)

class TestTopLogo10Test(unittest.TestCase):

    @property
    def dataset(self):
        """
        e.g.)
            tf = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            root_path = "/raid/projects/logo_detection/M2Det/datasets/toplogo10"
            dataset = TopLogo10s(root_path=root_path,transforms=tf).train
            return dataset
        """

        tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        root_path = "/raid/projects/logo_detection/M2Det/datasets/toplogo10"
        return TopLogo10s(root_path=root_path,transforms=tf).test

    def test_ds_object_has_returned_img_shape(self):
        ret_img, _ = self.dataset.__getitem__(0)

        logo_dim = self.dataset.IMG_DIM
        logo_shape = self.dataset.IMG_SHAPE

        expect_shape = (logo_dim, *logo_shape)

        self.assertEqual(expect_shape, ret_img.shape)

    def test_TopLogo10_returns_tensorized_img(self):
        ret_img, _ = self.dataset.__getitem__(0)
        self.assertTrue(isinstance(ret_img, torch.Tensor))

    def test_TopLogo10_returns_npndarrayized_target(self):
        _, target = self.dataset.__getitem__(0)
        self.assertTrue(isinstance(target, np.ndarray))

    def test_TopLogo10_returns_the_target_which_is_shape_of_Nx5(self):
        _, target = self.dataset.__getitem__(0)
        _, coords = target.shape
        self.assertEqual(coords, 5)






if __name__ == '__main__':
    unittest.main()
