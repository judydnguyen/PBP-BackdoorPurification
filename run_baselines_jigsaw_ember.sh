# ------------***----------------
# -------------APG---------------
# ------------***----------------

# cd src/train/jigsaw
# # python3 train_malimg_pytorch.py -c ../../../configs/backdoor/compare-rate-0.02.json
# CUDA_VISIBLE_DEVICES=1 python3 ft_ember_pytorch.py -c ../../../configs/backdoor/ember_train_0.005_ft_0.1.json
# CUDA_VISIBLE_DEVICES=1 python3 ft_jigsaw_pytorch.py -c ../../../configs/backdoor/apg_train_0.01_ft_0.1.json &&
# CUDA_VISIBLE_DEVICES=1 python3 ft_jigsaw_pytorch.py -c ../../../configs/backdoor/apg_train_0.02_ft_0.1.json &&
CUDA_VISIBLE_DEVICES=1 python3 src/train/jigsaw/ft_jigsaw_pytorch.py -c configs/backdoor/apg_train_0.05_ft_0.1.json

# CUDA_VISIBLE_DEVICES=1 python3 ft_jigsaw_pytorch.py -c ../../../configs/backdoor/apg_train_0.005_ft_0.05.json &&
# CUDA_VISIBLE_DEVICES=1 python3 ft_jigsaw_pytorch.py -c ../../../configs/backdoor/apg_train_0.01_ft_0.05.json &&
# CUDA_VISIBLE_DEVICES=1 python3 ft_jigsaw_pytorch.py -c ../../../configs/backdoor/apg_train_0.02_ft_0.05.json &&
# CUDA_VISIBLE_DEVICES=1 python3 ft_jigsaw_pytorch.py -c ../../../configs/backdoor/apg_train_0.05_ft_0.05.json
# cd ../../..