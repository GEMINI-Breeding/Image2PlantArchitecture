#!/bin/bash
nvidia-smi
# Script to run multiple experiments with different configurations
#TODAY_DATE=$(date +%Y%m%d)
TODAY_DATE="20250710_TrainOnFarm"
DATASET_PATH="/home/lion397/GEMINI/heesup/dataset/plant_architecture/20250311_Sideview_40Days"


# Create main log directory for all experiments
MAIN_LOG_DIR="log/${TODAY_DATE}"
mkdir -p $MAIN_LOG_DIR
echo "Created main log directory: $MAIN_LOG_DIR"
NUM_CPUS=$(nproc)

# Get number of available GPUs, but allow override
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
echo "Using $NUM_GPUS GPUs for training"

# Arrays of parameters
# PRELOAD="True"
PRELOAD="False"
#IMAGE_SIZES=(448 224)
IMAGE_SIZES=(224 448)
DEPTH_OPTIONS=("False")
SIDE_VIEWS=("True" "False")
ENCODERS=("facebook/dinov2-base" "facebook/dinov2-small")
DECODERS=("gpt2-medium" "gpt2")
#DECODERS=("google-bert/bert-base-uncased" "google-bert/bert-large-uncased")
#DECODERS=("gpt2" "gpt2-large")

EFFECTIVE_BATCH_SIZE=64
BATCH_SIZE=8   # Default is 4
GRAD_ACC=$((EFFECTIVE_BATCH_SIZE / BATCH_SIZE))
EPOCH=1

# Loop through all combinations (16 experiments total)
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
                    
                    # Create experiment name and directory
                    EXP_NAME="${ENCODER_NAME}_${IMAGE_SIZE}_${USE_DEPTH_STR}_${SIDE_VIEW_STR}_${DECODER_NAME}"
                    EXP_DIR="${MAIN_LOG_DIR}/${EXP_NAME}"
                    mkdir -p $EXP_DIR
                    
                    echo "Running experiment: $EXP_NAME"
                    echo "  Image Size: $IMAGE_SIZE"
                    echo "  RGB or Depth: $USE_DEPTH_STR"
                    echo "  Side View: $SIDE_VIEW"
                    echo "  Encoder: $ENCODER"
                    echo "  Decoder: $DECODER"
                    echo "  USE_DEPTH: $USE_DEPTH"
                    echo "  num_workers: $((NUM_CPUS/2))"
                    
                    # Run the experiment with dynamic GPU count
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
                            --num_workers $((NUM_CPUS/(NUM_GPUS*2))) \
                            --exp_name $EXP_NAME 2>&1 | tee -a "${EXP_DIR}/log.txt")
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
                            --num_workers $((NUM_CPUS/2)) \
                            --exp_name $EXP_NAME 2>&1 | tee -a "${EXP_DIR}/log.txt")
                    fi     
                    
                    echo "Experiment completed: $EXP_NAME"
                    echo "Logs saved to: ${EXP_DIR}/log.txt"
                    echo "Benchmark saved to: ${EXP_DIR}/benchmark.txt"
                    echo "-----------------------------------"
                done
            done
        done
    done
done

echo "All experiments completed!"
echo "Results saved to: $MAIN_LOG_DIR"

# Make sure to make the script executable with:
# chmod +x run_experiments.sh
