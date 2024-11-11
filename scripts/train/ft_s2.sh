export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=$(ifconfig | awk '/^[a-z]/ {gsub(/:/, ""); print $1; exit}')
export NCCL_DEBUG=WARN

PROMPT_VERSION="qwen_1_5"
LORA_ENABLE=True

VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
PREV_STAGE_CHECKPOINT=lmms-lab/llava-onevision-qwen2-0.5b-ov
# PREV_STAGE_CHECKPOINT="/dpc/kunf0097/.cache/huggingface/hub/v2-llava-qwen-ov-s1-1106_105645"
RUN_NAME="$( [[ "$LORA_ENABLE" == "True" ]] && echo "v2-lora-" || echo "v2-" )llava-qwen-ov-direct-$(date +%m%d_%H%M%S)"
# RUN_NAME="v2-lora-llava-qwen-ov-s1-1106_221039"

DATA_PATH=/home/kunet.ae/ku5001069/LLaVA-NeXT/data/s2/s2_train_v2.json
OUTPUT_DIR=/dpc/kunf0097/out/checkpoints/$RUN_NAME

echo "NCCl_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${RUN_NAME}"
echo "DATA_PATH: ${DATA_PATH}"

NUM_GPUS=$(nvidia-smi -L | wc -l)
NNODES=1
RANK=0
ADDR=$(hostname -I | awk '{print $1}')
PORT=29250

# CUDA_VISIBLE_DEVICES=0,2,3
# ACCELERATE_CPU_AFFINITY=1 accelerate launch --config_file /home/kunet.ae/ku5001069/LLaVA-NeXT/scripts/train/acc_cpu.yaml \
# deepspeed --master_port="${PORT}" \
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
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --verbose_logging True \
    --lora_enable ${LORA_ENABLE} \
    --lora_r 16 \
    --frames_upbound 32 \
    --fp16 True \
    # --bits 8
exit 0;