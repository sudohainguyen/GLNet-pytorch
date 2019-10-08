export CUDA_VISIBLE_DEVICES=0,1
python GLNet/tools/train.py \
--n_class 1 \
--data_path "../../datasets/paip2019/training_patches_2048" \
--model_path "../../checkpoints" \
--log_path "../../logs" \
--task_name "fpn_global_paip.512_lr2e5" \
--mode 1 \
--batch_size 32 \
--sub_batch_size 32 \
--size_g 512 \
--size_p 512 \
#--evaluation \
#--path_g "fpn_global.resize512_9.2.2018.2.global.pth" \
#--path_g2l "fpn_global2local.508.deep.cat.1x_ensemble_fmreg.p3_10.14.2018.lr2e5.pth" \
#--path_l2g "fpn_local2global.508_deep.cat_ensemble.p3_10.31.2018.lr2e5.local1x.epoch13.pth" \
# --path_g "cityscapes_global.800_4.5.2019.lr5e5.pth" \
# --path_g2l "fpn_global2local.508_deep.cat.1x_fmreg_ensemble.p3.0.15l2_3.19.2019.lr2e5.pth" \
# --path_l2g "fpn_local2global.508_deep.cat.1x_fmreg_ensemble.p3_3.19.2019.lr2e5.pth" \
