from torchvision.transforms import Compose
from pathlib import PosixPath

from m2det.datasets import preproc, VOCDetection, COCODetection
from m2det.datasets.toplogo10s.toplogo10s import TopLogo10s
from m2det.errors import Errors


class DataSets:
    NAME_TOPLOGO10 = TopLogo10s.NAME
    NAME_COCO = "COCO"
    NAME_VOC = "VOC"

    def __new__(cls, name, root_path: PosixPath or str, transforms: Compose, **kwargs):
        if kwargs:
            print("FutureWarning: ")

        if name == TopLogo10s.NAME:
            #TODO:
            _preproc = preproc(
                kwargs["cfg"].model.input_size,
                kwargs["cfg"].model.rgb_means,
                kwargs["cfg"].model.p
            )
            return TopLogo10s(
                root_path=root_path,
                # transforms=transforms
                transforms=_preproc
            )
        elif (name == cls.NAME_COCO) or (name == cls.NAME_VOC):
            return get_dataloader(cfg=kwargs["cfg"], dataset=kwargs["dataset"], setname=kwargs["setname"])
        else:
            msg = Errors.BASE_MSG
            msg += "NOT Supported datasets\n"
            msg += "\tname: {name}".format(name=name)
            raise ValueError(msg)


def get_dataloader(cfg, dataset, setname='train_sets'):
    _preproc = preproc(cfg.model.input_size, cfg.model.rgb_means, cfg.model.p)
    Dataloader_function = {'VOC': VOCDetection, 'COCO':COCODetection}
    _Dataloader_function = Dataloader_function[dataset]
    if setname == 'train_sets':
        dataset = _Dataloader_function(
            root=(cfg.COCOroot if dataset == 'COCO' else cfg.VOCroot),
            image_sets=getattr(cfg.dataset, dataset)[setname],
            preproc=_preproc
        )
    else:
        dataset = _Dataloader_function(cfg.COCOroot if dataset == 'COCO' else cfg.VOCroot,
                                   getattr(cfg.dataset, dataset)[setname], None)
    return dataset