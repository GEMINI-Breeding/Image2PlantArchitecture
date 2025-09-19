#!/bin/bash
nvidia-smi
# Script to run multiple experiments with different configurations
TODAY_DATE="20250713_TrainOnFarm"
DATASET_PATH="/home/lion397/GEMINI/heesup/dataset/plant_architecture/20250311_Sideview_40Days"

# Check node information
if [ -z "$NODE_ID" ]; then
    NODE_ID=0  # Default
fi

echo "Running on Node ID: $NODE_ID"

# Create main log directory for all experiments
MAIN_LOG_DIR="log/${TODAY_DATE}"
mkdir -p $MAIN_LOG_DIR
echo "Created main log directory: $MAIN_LOG_DIR"
NUM_CPUS=$(nproc)
NUM_WORKERS=$((NUM_CPUS/(NUM_GPUS*2)))

# Get number of available GPUs
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
echo "Using $NUM_GPUS GPUs for training"

# Arrays of parameters
PRELOAD="False"
IMAGE_SIZES=(224 448)
DEPTH_OPTIONS=("False")

# Distribute multi view training across nodes
if [ "$NODE_ID" -eq 0 ]; then
    SIDE_VIEWS=("True")    # Node 0: Side view
    echo "Node 0: Running Side view experiments only"
elif [ "$NODE_ID" -eq 1 ]; then
    SIDE_VIEWS=("False")   # Node 1: Top view
    echo "Node 1: Running Top view experiments only"
else
    SIDE_VIEWS=("True" "False") 
    echo "Other node: Running all view experiments"
fi

ENCODERS=("facebook/dinov2-base" "facebook/dinov2-small")
DECODERS=("gpt2-medium" "gpt2")

EFFECTIVE_BATCH_SIZE=64
BATCH_SIZE=8
GRAD_ACC=$((EFFECTIVE_BATCH_SIZE / BATCH_SIZE))
EPOCH=1

# Loop through all combinations
for IMAGE_SIZE in "${IMAGE_SIZES[@]}"; do
    for SIDE_VIEW in "${SIDE_VIEWS[@]}"; do
        for ENCODER in "${ENCODERS[@]}"; do
            for DECODER in "${DECODERS[@]}"; do
                for USE_DEPTH in "${DEPTH_OPTIONS[@]}"; do
                    # Extract model names for directory naming
                    ENCODER_NAME=$(echo $ENCODER | cut -d'/' -f2)
                    DECODER_NAME=$(echo $DECODER | cut -d'/' -f2)
                    SIDE_VIEW_STR=$(if [ "$SIDE_VIEW" = "True" ]; then echo "Sideview"; else echo "TopView"; fi)
                    USE_DEPTH_STR=$(if [ "$USE_DEPTH" = "True" ]; then echo "Depth"; else echo "RGB"; fi)
                    
                    # Create experiment name with node info
                    EXP_NAME="${ENCODER_NAME}_${IMAGE_SIZE}_${USE_DEPTH_STR}_${SIDE_VIEW_STR}_${DECODER_NAME}"
                    EXP_DIR="${MAIN_LOG_DIR}/${EXP_NAME}"
                    mkdir -p $EXP_DIR
                    
                    echo "Node ${NODE_ID} - Running experiment: $EXP_NAME"
                    echo "  Image Size: $IMAGE_SIZE"
                    echo "  RGB or Depth: $USE_DEPTH_STR"
                    echo "  Side View: $SIDE_VIEW"
                    echo "  Encoder: $ENCODER"
                    echo "  Decoder: $DECODER"
                    echo "  USE_DEPTH: $USE_DEPTH"
                    echo "  num_workers: $NUM_WORKERS"
                    echo "  Effective Batch Size: $EFFECTIVE_BATCH_SIZE"
                    echo "  Batch Size: $BATCH_SIZE"
                    echo "  Gradient Accumulation: $GRAD_ACC"
                    echo "  Epochs: $EPOCH"
                    
                    # Run the experiment (Use one gpu per node)
                    if [ "$NUM_GPUS" -gt 1 ]; then
                        (accelerate launch --multi_gpu --num_processes=$NUM_GPUS src/train_hf.py \
                            --image_size $IMAGE_SIZE \
                            --side_view $SIDE_VIEW \
                            --preload $PRELOAD \
                            --encoder_checkpoint $ENCODER \
                            --decoder_checkpoint $DECODER \
                            --dataset_path $DATASET_PATH \
                            --today_date_str $TODAY_DATE \
                            --epoch $EPOCH \
                            --batch_size $BATCH_SIZE \
                            --grad_acc $GRAD_ACC \
                            --use_depth $USE_DEPTH \
                            --num_workers $NUM_WORKERS \
                            --exp_name $EXP_NAME 2>&1 | tee -a "${EXP_DIR}/log_node${NODE_ID}.txt")
                    else
                        (python src/train_hf.py \
                            --image_size $IMAGE_SIZE \
                            --side_view $SIDE_VIEW \
                            --preload $PRELOAD \
                            --encoder_checkpoint $ENCODER \
                            --decoder_checkpoint $DECODER \
                            --dataset_path $DATASET_PATH \
                            --today_date_str $TODAY_DATE \
                            --epoch $EPOCH \
                            --batch_size $BATCH_SIZE \
                            --grad_acc $GRAD_ACC \
                            --use_depth $USE_DEPTH \
                            --num_workers $NUM_WORKERS \
                            --exp_name $EXP_NAME 2>&1 | tee -a "${EXP_DIR}/log_node${NODE_ID}.txt")
                    fi     
                    
                    echo "Node ${NODE_ID} - Experiment completed: $EXP_NAME"
                    echo "Logs saved to: ${EXP_DIR}/log_node${NODE_ID}.txt"
                    echo "-----------------------------------"
                done
            done
        done
    done
done

echo "Node ${NODE_ID} - All experiments completed!"
echo "Results saved to: $MAIN_LOG_DIR"