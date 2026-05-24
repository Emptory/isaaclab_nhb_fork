# import task
from .tasks import *

import os
ISAACLAB_NHB_PATH = os.path.dirname(os.path.abspath(__file__))
ISAACLAB_NHB_ASSET_PATH = os.path.join(ISAACLAB_NHB_PATH, 'assets/')
ISAACLAB_ROBOT_DESCRIPTION_PATH = os.path.join(ISAACLAB_NHB_PATH, '../..', 'robot_description')
ISAACLAB_NHB_AMP_DATA_PATH = os.path.join(ISAACLAB_NHB_PATH, 'tasks/amp_data_cfg/XDdataset')


# 下面是全局变量
# 是否打开图形渲染
HEADLESS_FLAG = False

