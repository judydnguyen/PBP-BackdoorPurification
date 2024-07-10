
# -------- EMBER DATASET -------- #
# # cd src/train/ember
# CUDA_VISIBLE_DEVICES=1 python3 train_ember_pytorch.py -c configs/backdoor/ember_train_default.json
# CUDA_VISIBLE_DEVICES=1 python3 train_ember_pytorch.py -c configs/backdoor/ember_train_0.01_ft_0.1.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_ember_pytorch.py -c configs/backdoor/ember_train_0.02_ft_0.1.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_ember_pytorch.py -c configs/backdoor/ember_train_0.05_ft_0.1.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_ember_pytorch.py -c configs/backdoor/ember_train_0.005_ft_0.1.json
# CUDA_VISIBLE_DEVICES=1 python3 train_ember_pytorch.py -c configs/backdoor/ember_train_0.005_ft_0.05.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_ember_pytorch.py -c configs/backdoor/ember_train_0.01_ft_0.05.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_ember_pytorch.py -c configs/backdoor/ember_train_0.02_ft_0.05.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_ember_pytorch.py -c configs/backdoor/ember_train_0.05_ft_0.05.json 
# # cd ../../..
# # -------- ************** -------- #
# Generate backdoor data if you havent done so
# CUDA_VISIBLE_DEVICES=1 python3 src/train/ember/generate_backdoor_data.py -c configs/backdoor/ember_train_default.json
# CUDA_VISIBLE_DEVICES=1 python3 src/train/ember/train_ember_pytorch.py -c configs/backdoor/ember_train_default.json
# # -------- ************** -------- #


# # -------- JIGSAW DATASET -------- #
# # -------- ************** -------- #
# # cd src/train/jigsaw
# CUDA_VISIBLE_DEVICES=1 python3 train_ember_pytorch.py -c configs/backdoor/ember_train_default.json
# CUDA_VISIBLE_DEVICES=0 python3 train_jigsaw_pytorch.py -c configs/train/apg_train_default_ft_0.1.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/backdoor/apg_train_0.01_ft_0.1.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/backdoor/apg_train_0.005_ft_0.1.json &&
CUDA_VISIBLE_DEVICES=1 python3 src/train/jigsaw/train_jigsaw_pytorch.py -c configs/backdoor/apg_train_0.05_ft_0.1.json
# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/backdoor/apg_train_0.02_ft_0.1.json && 

# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/backdoor/apg_train_0.01_ft_0.05.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/backdoor/apg_train_0.005_ft_0.05.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/backdoor/apg_train_0.02_ft_0.05.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/backdoor/apg_train_0.05_ft_0.05.json
# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/train/apg_train_0.01_ft_0.05.json
# cd ../../..
