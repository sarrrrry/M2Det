from pathlib import PosixPath, Path

from abc import abstractmethod


class BaseDataSets:
    def __init__(self, root_path: PosixPath or str, transforms=None):

        if root_path:  # meaning: if root_path is not ""(empty string)
            root_path = Path(root_path)
            if not root_path.exists():
                msg = str(
                    "Error: NOT Exists: {}".format(root_path)
                )
        self.root_path = root_path
        self.transforms = transforms

    @property
    @abstractmethod
    def train(self):
        pass

    @property
    @abstractmethod
    def test(self):
        pass

    @property
    @abstractmethod
    def val(self):
        pass

    @property
    @abstractmethod
    def NAME(self):
        pass
