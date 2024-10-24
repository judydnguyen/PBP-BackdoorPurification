#!/bin/bash
FOLDER_PATH="configs/r2-ndss/change_class_ratio/ember"

# Iterate over each file in the configuration directory
for config in "$FOLDER_PATH"/*; do
    if [[ -f $config ]]; then
        # Run the Python script with the current configuration file
        CUDA_VISIBLE_DEVICES=1 python src/train/ember/ft_ember_pytorch.py -c "$config"
    fi
done