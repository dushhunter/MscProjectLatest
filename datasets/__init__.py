# StoneVol_main vs SPIdepth-main: adds StoneDataset and related dataset wiring.
from .nyu_raw_dataset import NYUrawDataset
from .mc_dataset import *
from .mono_dataset_mc import *
from .kitti_dataset import *
from .mono_dataset import *
from .mono_dataset_city import *
from .cityscapes_preprocessed_dataset import CityscapesPreprocessedDataset
from .cityscapes_evaldataset import CityscapesEvalDataset
from .stone_dataset import StoneDataset  # StoneVol_main: adds stone dataset loader