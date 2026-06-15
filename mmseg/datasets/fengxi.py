from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class fengxiDataset(BaseSegDataset):
    """gas Potsdam dataset.

     In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
     ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
     ``seg_map_suffix`` are both fixed to '.png'.
     """
    METAINFO = dict(
        classes=['background', 'fengxi'],  # 显式添加背景类别
        palette=[[255, 255, 255], [0, 0, 0]])  # 背景为白色，气体为黑色（reduce_zero_label=False）



    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)