# CUDA_VISIBLE_DEVICES=0 python3 src/train/jigsaw/ft_jigsaw_pytorch.py -c configs/r2-ndss/change_family_ratio/apg_train_0.05_ft_0.1_param_0.1.json
# CUDA_VISIBLE_DEVICES=0 python3 src/train/ember/ft_ember_pytorch.py -c configs/r2-ndss/change_class_ratio/ember/ember_train_0.05_ft_0.1_param_0.2.json
# CUDA_VISIBLE_DEVICES=0 python3 src/train/ember/ft_ember_pytorch.py -c configs/r2-ndss/change_overlapping_ratio/ember/ember_train_0.05_ft_0.1_param_1.0.json
# CUDA_VISIBLE_DEVICES=0 python3 src/train/jigsaw/ft_jigsaw_pytorch.py -c configs/r2-ndss/change_mask_ratio/apg_train_0.01_ft_0.01_param_0.001.json
# CUDA_VISIBLE_DEVICES=0 python3 src/train/ember/ft_ember_pytorch.py -c configs/r2-ndss/change_mask_ratio/ember/ember_train_0.05_ft_0.1_param_0.001.json
CUDA_VISIBLE_DEVICES=0 python3 src/train/jigsaw/ft_jigsaw_pytorch.py -c configs/r2-ndss/change_family_ratio/apg_train_0.05_ft_0.1_param_0.01.json