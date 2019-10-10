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
export CUDA_VISIBLE_DEVICES=0,1
python GLNet/tools/train.py \
--n_class 2 \
--data_path "../../datasets/paip2019/training_patches_4096" \
--model_path "../../checkpoints" \
--log_path "../../logs" \
--task_name "fpn_mode1_paip.1008_lr2e5" \
--mode 1 \
--batch_size 8 \
--sub_batch_size 32 \
--size_g 1008 \
--size_p 1008 \
--num_workers 10
