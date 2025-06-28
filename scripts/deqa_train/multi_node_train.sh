#!/bin/bash
# Minimal Multi-Node Distributed Training Script
# Usage: ./simple_multi_node_train.sh

# ============================================
# BASIC CONFIGURATION
# ============================================

# Node configuration
NODES=(
    "xx.xx.xx.xx"
    "xx.xx.xx.xx" 
)

MASTER_NODE="xx.xx.xx.xx"
NNODES=${#NODES[@]}
GPUS_PER_NODE=8

# Training parameters
MODEL_PATH="Qwen/Qwen2.5-VL-32B-Instruct"
OUTPUT_DIR="./output"

# Define experiments
EXPERIMENTS=(
    "overall_quality:./data/train_file_v2/train_diqa_overall_quality.json:$OUTPUT_DIR/overall_quality"
    "sharpness:./data/train_file_v2/train_diqa_sharpness.json:$OUTPUT_DIR/sharpness"
    "color_fidelity:./data/train_file_v2/train_diqa_color_fidelity.json:$OUTPUT_DIR/color_fidelity"
)

echo "MODEL_PATH: $MODEL_PATH"
echo "Total experiments: ${#EXPERIMENTS[@]}"

# ============================================
# HELPER FUNCTIONS
# ============================================

# Find free port on master node
find_free_port() {
    echo "🔎 Finding free port on $MASTER_NODE..."
    MASTER_PORT=$(ssh -o ConnectTimeout=10 "$MASTER_NODE" "python -c 'import socket; s = socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1]); s.close()'")
    
    if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]]; then
        echo "❌ Failed to get free port"
        exit 1
    fi
    
    echo "✅ Using port: $MASTER_PORT"
}

# Setup environment
setup_env() {
    echo "🔧 Setting up environment..."
    
    # Find free port
    find_free_port
    
    # Set NCCL environment
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


    # Set DeepSpeed environment
    export MASTER_ADDR=$MASTER_NODE
    export MASTER_PORT=$MASTER_PORT
    export WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
    
    # Set Python path
    export PYTHONPATH=src:$PYTHONPATH
    
    echo "✅ Environment ready - Master: $MASTER_ADDR:$MASTER_PORT, World size: $WORLD_SIZE"
}

# Create hostfile
create_hostfile() {
    echo "📝 Creating hostfile..."
    HOSTFILE="hostfile_simple"
    > "$HOSTFILE"
    
    for node in "${NODES[@]}"; do
        echo "${node} slots=${GPUS_PER_NODE}" >> "$HOSTFILE"
    done
    
    echo "✅ Hostfile created"
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
    
    # Run training
    echo "🏃 Launching DeepSpeed training for $experiment_name..."
    export WANDB_PROJECT="DeQA-Score"

    deepspeed --hostfile="$HOSTFILE" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        src/train/train_deqa.py \
        --deepspeed scripts/zero3.json \
        --model_id "$MODEL_PATH" \
        --data_path "$data_path" \
        --res_image_folder "./data/DIQA/train/res" \
        --ori_image_folder "./data/DIQA/train/ori" \
        --output_dir "$output_dir" \
        --num_train_epochs 4 \
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
        --save_steps 100 \
        --save_total_limit 5 \
        --gradient_checkpointing True \
        --bf16 True \
        --dataloader_num_workers 16 \
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
        --run_name $experiment_name \
        --use_ori_image True

    
    echo "✅ Experiment $experiment_name completed!"
}

# ============================================
# MAIN TRAINING
# ============================================

main() {
    echo "🚀 Starting sequential multi-node training for all experiments..."
    echo "Nodes: ${NODES[*]}"
    echo "Model: $MODEL_PATH"
    
    # Setup
    setup_env
    create_hostfile
    
    # Run all experiments sequentially
    for experiment in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r experiment_name data_path output_dir <<< "$experiment"
        
        echo "============================================"
        echo "Starting experiment: $experiment_name"
        echo "============================================"
        
        run_experiment "$experiment_name" "$data_path" "$output_dir"
        
        echo "============================================"
        echo "Completed experiment: $experiment_name"
        echo "============================================"
        echo ""
    done
    
    echo "🎉 All experiments completed successfully!"
    
    # Cleanup
    rm -f "$HOSTFILE"
}

# Run main function
main "$@" 