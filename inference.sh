export CUDA_VISIBLE_DEVICES=0
python GLNet/tools/inference.py \
--n_class 2 \
--set training \
--path_g ../../checkpoints/fpn.pth \
--path_l2g ../../checkpoints/fpn.pth \
--img_idx 01_01_0085 \
--save_pred
