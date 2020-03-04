export CUDA_VISIBLE_DEVICES=0
python GLNet/tools/evaluate.py \
--n_class 2 \
--data_path "../../datasets/paip2019/downsampled_training" \
--model_path "../../checkpoints" \
--log_path "../../logs" \
--task_name "fpn_mode2_paipdown4_244_preprocessing_opening" \
--mode 2 \
--batch_size 16 \
--sub_batch_size 16 \
--size_g 244 \
--size_p 244 \
--num_workers 5 \
--gpu_ids 0 \
--path_g "fpn_mode1_paipdown4_244_preprocessing_opening.pth" \
--path_g2l "fpn_mode2_paipdown4_244_preprocessing_opening.pth"
# --path_l2g "fpn_mode3_paip.1008_lr5e5.pth"
# --save_predictions "../../gl_pred_mode1" \
