# export CUDA_VISIBLE_DEVICES=0,1
# python GLNet/tools/train.py \
# --n_class 2 \
# --data_path "../../datasets/paip2019/training_patches_2048" \
# --model_path "../../checkpoints" \
# --log_path "../../logs" \
# --task_name "fpn_mode1_paip.500_lr2e5" \
# --mode 1 \
# --batch_size 32 \
# --sub_batch_size 32 \
# --size_g 500 \
# --size_p 500 \
export CUDA_VISIBLE_DEVICES=1
python GLNet/tools/train.py \
--n_class 2 \
--data_path "../../datasets/paip2019/downsampled_training" \
--model_path "../../checkpoints" \
--log_path "../../logs" \
--task_name "fpn_mode2_paipdown4_244_preproc_opening_rmsmall" \
--lr 5e-5 \
--mode 2 \
--batch_size 32 \
--sub_batch_size 32 \
--size_g 244 \
--size_p 244 \
--num_workers 5 \
--path_g "fpn_mode1_paipdown4_244_preproc_opening_rmsmall.pth" \
# --path_g2l "fpn_mode1_paipdown4_244_preprocessing_opening.pth"
