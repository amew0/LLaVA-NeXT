export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eno1 # get it from ifconfig (that matches hostname -I)
export NCCL_DEBUG=INFO

# LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
# LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
# VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

# BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
# echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-ov-$(date +%m%d_%H%M%S)" 
# PREV_STAGE_CHECKPOINT="/mnt/bn/vl-research/checkpoints/onevision/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mid_to_final_next_3m_am9_july14" # replace it with your last checkpoint training from single image collection
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-0.5b-ov"
DATA_PATH=/home/kunet.ae/ku5001069/LLaVA-NeXT/data/train_4.json
OUTPUT_DIR=/dpc/kunf0097/out/checkpoints/$RUN_NAME
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"
echo "DATA_PATH: ${DATA_PATH}"

NUM_GPUS=$(nvidia-smi -L | wc -l)
NNODES=1
RANK=0
ADDR=$(hostname -I | awk '{print $1}')
PORT=29500


ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --cache_dir /dpc/kunf0097/cache/models \
    --version $PROMPT_VERSION \
    --data_path ${DATA_PATH} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 False \
    --run_name $RUN_NAME \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --verbose_logging True \
    --fp16 True \
    --lora_enable True \
    --frames_upbound 32
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
