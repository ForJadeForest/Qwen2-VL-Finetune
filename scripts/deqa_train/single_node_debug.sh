#!/bin/bash

# Single-Node Debug Training Script
# Usage: ./single_node_debug.sh [experiment_name]

# ============================================
# BASIC CONFIGURATION
# ============================================

# Training parameters
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
GPUS_PER_NODE=1

# Define experiments
declare -A EXPERIMENTS
EXPERIMENTS=(
    ["overall_quality"]="./data/train_file_v2/train_diqa_overall_quality.json:./debug_results/overall_quality"
    ["sharpness"]="./data/train_file_v2/train_diqa_sharpness.json:./debug_results/sharpness"
    ["color_fidelity"]="./data/train_file_v2/train_diqa_color_fidelity.json:./debug_results/color_fidelity"
)

# Default experiment if none specified
DEFAULT_EXPERIMENT="overall_quality"

echo "MODEL_PATH: $MODEL_PATH"
echo "Available experiments: ${!EXPERIMENTS[@]}"

# ============================================
# HELPER FUNCTIONS
# ============================================

# Find free port for distributed training
find_free_port() {
    echo "🔎 Finding free port..."
    MASTER_PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    
    if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]]; then
        echo "❌ Failed to get free port, using default 29500"
        MASTER_PORT=29500
    fi
    
    echo "✅ Using port: $MASTER_PORT"
}

# Setup environment
setup_env() {
    echo "🔧 Setting up environment..."
    
    # Find free port
    find_free_port
    
    # Set NCCL environment for multi-GPU training
    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=bond1
    export NCCL_TREE_THRESHOLD=0
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECKS_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=0


    # Set distributed training environment for single node
    export MASTER_ADDR=localhost
    export MASTER_PORT=$MASTER_PORT
    export WORLD_SIZE=$GPUS_PER_NODE
    export RANK=0
    export LOCAL_RANK=0
    
    # Set Python path
    export PYTHONPATH=src:$PYTHONPATH
    
    echo "✅ Environment ready - Master: $MASTER_ADDR:$MASTER_PORT, World size: $WORLD_SIZE"
}

# Run single experiment
run_experiment() {
    local experiment_name=$1
    local data_path=$2
    local output_dir=$3
    
    echo "🚀 Starting experiment: $experiment_name"
    echo "Data: $data_path"
    echo "Output: $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run training with DeepSpeed
    echo "🏃 Launching single-node DeepSpeed training for $experiment_name..."
    export WANDB_PROJECT="DeQA-Score-Debug"


    # Use deepspeed launcher for single node
    deepspeed --num_gpus=$GPUS_PER_NODE \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        src/train/train_deqa.py \
        --deepspeed scripts/zero2.json \
        --model_id "$MODEL_PATH" \
        --data_path "$data_path" \
        --res_image_folder "./data/DIQA/train/res" \
        --ori_image_folder "./data/DIQA/train/ori" \
        --output_dir "$output_dir" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
        --vision_lr 1e-5 \
        --merger_lr 1e-5 \
        --weight_decay 0.1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --save_strategy steps \
        --save_steps 50 \
        --save_total_limit 3 \
        --gradient_checkpointing True \
        --bf16 True \
        --dataloader_num_workers 8 \
        --remove_unused_columns False \
        --freeze_vision_tower True \
        --freeze_llm False \
        --freeze_merger True \
        --disable_flash_attn2 False \
        --level_prefix "The quality of the image is" \
        --level_names great good fair weak bad \
        --use_softkl_loss True \
        --max_grad_norm 1 \
        --save_only_model \
        --use_liger False \
        --use_ori_image True \
        --run_name "${experiment_name}_debug" \
    
    echo "✅ Experiment $experiment_name completed!"
}

# ============================================
# MAIN TRAINING
# ============================================

main() {
    local experiment_name=${1:-$DEFAULT_EXPERIMENT}
    
    # Check if experiment exists
    if [[ ! -v EXPERIMENTS[$experiment_name] ]]; then
        echo "❌ Invalid experiment name: $experiment_name"
        echo "Available experiments: ${!EXPERIMENTS[@]}"
        exit 1
    fi
    
    # Parse experiment configuration
    IFS=':' read -r data_path output_dir <<< "${EXPERIMENTS[$experiment_name]}"
    
    echo "🚀 Starting single-node debug training..."
    echo "Experiment: $experiment_name"
    echo "Model: $MODEL_PATH"
    echo "Data: $data_path"
    echo "Output: $output_dir"
    
    # Setup
    setup_env
    
    # Run experiment
    echo "============================================"
    echo "Starting debug training: $experiment_name"
    echo "============================================"
    
    run_experiment "$experiment_name" "$data_path" "$output_dir"
    
    echo "============================================"
    echo "Completed debug training: $experiment_name"
    echo "============================================"
    
    echo "🎉 Debug training completed successfully!"
}

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [experiment_name]"
    echo "Available experiments: ${!EXPERIMENTS[@]}"
    echo "Using default experiment: $DEFAULT_EXPERIMENT"
    echo ""
fi

# Run main function
main "$@" 