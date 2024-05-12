"""
Alif trying to understand the scripts

"""

import numpy as np
import pandas as pd
import mc
import rsatoolbox


# SELECT PARTICIPANT / TASK HALF
SUB_NO = '01'
 
TASK_HALF = 1
# path to data
data_dir_beh = "/Users/student/PycharmProjects/data"
path_beh = f"{data_dir_beh}/raw/sub-{SUB_NO}/beh"
sub_fmri_beh = f"{path_beh}/sub-{SUB_NO}_fmri_pt{TASK_HALF}.csv"
