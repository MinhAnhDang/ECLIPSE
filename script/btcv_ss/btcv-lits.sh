#!/bin/bash

ngpus=$(nvidia-smi --list-gpus | wc -l)

cfg_file=configs/btcv/maskformer2_R50_bs16_160k.yaml
base=results/btcv_ss
step_args="CONT.BASE_CLS 13 CONT.INC_CLS 2 CONT.MODE overlap SEED 42"
task=btcv_13-2-ov

name=MxF
meth_args="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.FOCAL True"

base_queries=13
dice_weight=5.0
mask_weight=5.0
class_weight=2.0

base_lr=0.0001
iter=5000

soft_mask=False # mask softmax (True) or sigmoid (False)
soft_cls=False   # classifier softmax (True) or sigmoid( False)

num_prompts=0
deep_cls=True

weight_args="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${base_queries} MODEL.MASK_FORMER.DICE_WEIGHT ${dice_weight} MODEL.MASK_FORMER.MASK_WEIGHT ${mask_weight} MODEL.MASK_FORMER.CLASS_WEIGHT ${class_weight} MODEL.MASK_FORMER.SOFTMASK ${soft_mask} CONT.SOFTCLS ${soft_cls} CONT.NUM_PROMPTS ${num_prompts} CONT.DEEP_CLS ${deep_cls}"

exp_name="btcv_13_2"

comm_args="OUTPUT_DIR ${base} ${meth_args} ${step_args} ${weight_args}"
inc_args="CONT.TASK 0 SOLVER.BASE_LR ${base_lr} TEST.EVAL_PERIOD 500 SOLVER.CHECKPOINT_PERIOD 500 SOLVER.MAX_ITER ${iter}"

## Train base classes
## You can skip this process if you have a step0-checkpoint.
python train_inc.py --num-gpus ${ngpus} --config-file ${cfg_file} ${comm_args} ${inc_args} NAME ${exp_name} WANDB False


# --------------------------------------
# Training continual on LiTS dataset
# cfg_file=configs/btcv/maskformer2_R50_bs16_160k.yaml
# base=results/btcv_ss
# base_queries=13
# num_prompts=2

# iter=1200
# base_lr=0.001

# dice_weight=5.0
# mask_weight=5.0
# class_weight=10.0

# backbone_freeze=True
# trans_decoder_freeze=True
# pixel_decoder_freeze=True
# cls_head_freeze=True
# mask_head_freeze=True
# query_embed_freeze=True

# prompt_deep=True
# prompt_mask_mlp=True
# prompt_no_obj_mlp=False

# deltas=[-0.3,-0.6]
# deep_cls=True

# weight_args="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${base_queries} MODEL.MASK_FORMER.DICE_WEIGHT ${dice_weight} MODEL.MASK_FORMER.MASK_WEIGHT ${mask_weight} MODEL.MASK_FORMER.CLASS_WEIGHT ${class_weight} MODEL.MASK_FORMER.SOFTMASK ${soft_mask} CONT.SOFTCLS ${soft_cls} CONT.NUM_PROMPTS ${num_prompts}"
# comm_args="OUTPUT_DIR ${base} ${meth_args} ${step_args} ${weight_args}"

# inc_args="CONT.TASK 1 SOLVER.MAX_ITER ${iter} SOLVER.BASE_LR ${base_lr} TEST.EVAL_PERIOD 4000 SOLVER.CHECKPOINT_PERIOD 500000 CONT.WEIGHTS results/ade_ss_100_step0.pth"

# vpt_args="CONT.BACKBONE_FREEZE ${backbone_freeze} CONT.CLS_HEAD_FREEZE ${cls_head_freeze} CONT.MASK_HEAD_FREEZE ${mask_head_freeze} CONT.PIXEL_DECODER_FREEZE ${pixel_decoder_freeze} CONT.QUERY_EMBED_FREEZE ${query_embed_freeze} CONT.TRANS_DECODER_FREEZE ${trans_decoder_freeze} CONT.PROMPT_MASK_MLP ${prompt_mask_mlp} CONT.PROMPT_NO_OBJ_MLP ${prompt_no_obj_mlp} CONT.PROMPT_DEEP ${prompt_deep} CONT.DEEP_CLS ${deep_cls} CONT.LOGIT_MANI_DELTAS ${deltas}"

# exp_name="btcv_13_2"

# python train_inc.py --num-gpus ${ngpus} --config-file ${cfg_file} ${comm_args} ${inc_args} ${cont_args} ${dist_args} ${vpt_args} NAME ${exp_name} WANDB False


# # -------- evaluation ------------------------------

# deltas=[0.0,-0.1]

# vpt_args="CONT.BACKBONE_FREEZE ${backbone_freeze} CONT.CLS_HEAD_FREEZE ${cls_head_freeze} CONT.MASK_HEAD_FREEZE ${mask_head_freeze} CONT.PIXEL_DECODER_FREEZE ${pixel_decoder_freeze} CONT.QUERY_EMBED_FREEZE ${query_embed_freeze} CONT.TRANS_DECODER_FREEZE ${trans_decoder_freeze} CONT.PROMPT_MASK_MLP ${prompt_mask_mlp} CONT.PROMPT_NO_OBJ_MLP ${prompt_no_obj_mlp} CONT.PROMPT_DEEP ${prompt_deep} CONT.DEEP_CLS ${deep_cls} CONT.LOGIT_MANI_DELTAS ${deltas}"

# inc_args="CONT.TASK 1 CONT.WEIGHTS results/ade_ss_100_50_final.pth"

# python train_inc.py --eval-only --num-gpus ${ngpus} --config-file ${cfg_file} ${comm_args} ${inc_args} ${cont_args} ${dist_args} ${vpt_args} NAME ${exp_name} WANDB False
