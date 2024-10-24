#!/bin/bash
# a bash file to change the overlapping ratio of the R2 reads
# loop through a folder path and change the config file

# FOLDER_PATH="configs/r2-ndss/change_mask_ratio"
FOLDER_PATH="configs/r2-ndss/change_mask_ratio"

# Iterate over each file in the configuration directory
for config in "$FOLDER_PATH"/*; do
    if [[ -f $config ]]; then
        # Run the Python script with the current configuration file
        CUDA_VISIBLE_DEVICES=0 python src/train/jigsaw/ft_jigsaw_pytorch.py -c "$config"
    fi
done