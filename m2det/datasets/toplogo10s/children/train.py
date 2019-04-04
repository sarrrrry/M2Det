from m2det.datasets.toplogo10s.children.base_toplogo10_child_dataset import BaseTL10ChildDataSet
from m2det.errors import Errors


class TL10Train(BaseTL10ChildDataSet):
    @property
    def name_list(self):
        name_list_path = self.data_name_root/"60_images_per_class_train.txt"
        if not name_list_path.exists():
            raise Errors().FileNotFound(name_list_path)

        with name_list_path.open() as f:
            train_name_list = f.read()
            train_name_list = train_name_list.split("\n")
        return train_name_list