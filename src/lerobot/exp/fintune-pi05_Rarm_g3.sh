#!/bin/bash
# Pi0.5 训练脚本（与 create_g2_dataset_using_lerobot 导出数据 G1-7 配套）
# 需在 **lerobot_challenge 仓库根目录** 下执行（与你们环境一致），以便 ./ckpts、accelerate、lerobot-train 可用。
# 使用前请修改下方 DATA_PATH、NUM_GPUS、PRETRAINED_PATH 等。

set -euo pipefail

# HF 镜像与代理
export HF_ENDPOINT=https://hf-mirror.com
# unset all_proxy ALL_PROXY HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
# ==================== 配置区域 ====================
NUM_GPUS=6               # GPU 数量，改成你的卡数
BATCH_SIZE=64
CHUNK_SIZE=45              # 与配置的 chunk_size / n_action_steps 对齐 1.5s * 30 FPS
PRETRAINED_PATH=lerobot/pi05_base #可修改为本地路径
RESUME=false              # 若要从中断继续，置为 true 并填写 RESUME_CONFIG_PATH
RESUME_CONFIG_PATH=null
JOB_NAME=pi05_g2_right_arm_g3
# 导出得到的 LeRobot 数据集根目录（port_phys_twin.py 的 --save-path）
DATA_PATH=/data1/training_data/lerobot_exports/g2_right_arm_g3
DATASET_REPO_ID=g2_right_arm_g3
#输出保存目录
OUTPUT_PATH=/data2/exp/output_pi05_g2_right_arm_g3
# 相对动作
USE_RELATIVE_ACTIONS=false
# ==================== 配置区域 ====================

# 指定使用的 GPU（根据你的实际情况修改编号）
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# 多 GPU 训练
accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  $(which lerobot-train) \
  --dataset.root=${DATA_PATH} \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.video_backend=torchcodec \
  --num_workers=24 \
  --batch_size=${BATCH_SIZE} \
  --policy.device=cuda \
  --policy.path=${PRETRAINED_PATH} \
  --rename_map='{"observation.images.head_color":"observation.images.base_0_rgb", "observation.images.hand_right_color":"observation.images.right_wrist_0_rgb"}' \
  --policy.chunk_size=${CHUNK_SIZE} \
  --policy.n_action_steps=${CHUNK_SIZE} \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --policy.compile_model=true \
  --policy.use_relative_actions=${USE_RELATIVE_ACTIONS} \
  --job_name=${JOB_NAME} \
  --eval_freq=5000 \
  --save_freq=2500 \
  --log_freq=1000 \
  --steps=25000 \
  --policy.gradient_checkpointing=true \
  --output_dir=${OUTPUT_PATH}

# 如果需要恢复训练，取消下面的注释并设置 RESUME=true 和 RESUME_CONFIG_PATH
  # --resume=${RESUME} \
  # --resume_config_path=${RESUME_CONFIG_PATH} \
