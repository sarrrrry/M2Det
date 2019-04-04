from m2det.datasets.base_datasets import BaseDataSets
from m2det.datasets.toplogo10s.children.test import TL10Test
from m2det.datasets.toplogo10s.children.train import TL10Train


class TopLogo10s(BaseDataSets):
    NAME = "toplogo10"

    @property
    def train(self):
        return TL10Train(self)

    @property
    def test(self):
        return TL10Test(self)

    @property
    def val(self):
        raise AttributeError("val property will develop in future")


