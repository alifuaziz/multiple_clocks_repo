#!bin/bash

module load fsl

for subTag in 01 02 03 04 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35 

    # Run the script for all the subjects
    fsl_sub -T 100 /home/fs0/chx061/.conda/envs/rsa/bin/python /home/fs0/chx061/multiple_clocks_repo/mc/replay_analysis/scripts/step_wise_group_analysis/step_wise_group_analysis.py sub-${subTag}

done