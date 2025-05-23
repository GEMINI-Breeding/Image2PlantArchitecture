#!/bin/bash

# Script to run multiple experiments with different configurations
#TODAY_DATE=$(date +%Y%m%d)
TODAY_DATE="20250523_TrainOnFarm"
# DATASET_PATH="data/2000_Plots_20241210_BetterQuantized"


# Create main log directory for all experiments
MAIN_LOG_DIR="log/${TODAY_DATE}"
mkdir -p $MAIN_LOG_DIR
echo "Created main log directory: $MAIN_LOG_DIR"

# Arrays of parameters
IMAGE_SIZES=(224 448)
EPOCH=8
SIDE_VIEWS=("True" "False")
ENCODERS=("facebook/dinov2-small" "facebook/dinov2-base")
#DECODERS=("google-bert/bert-base-uncased" "google-bert/bert-large-uncased")
#DECODERS=("gpt2" "gpt2-large")
DECODERS=("gpt2" "gpt2-medium")

# Loop through all combinations (16 experiments total)
for IMAGE_SIZE in "${IMAGE_SIZES[@]}"; do
    for SIDE_VIEW in "${SIDE_VIEWS[@]}"; do
        for ENCODER in "${ENCODERS[@]}"; do
            for DECODER in "${DECODERS[@]}"; do
                # Extract model names for directory naming
                ENCODER_NAME=$(echo $ENCODER | cut -d'/' -f2)
                DECODER_NAME=$(echo $DECODER | cut -d'/' -f2)
                SIDE_VIEW_STR=$(if [ "$SIDE_VIEW" = "True" ]; then echo "Sideview"; else echo "TopView"; fi)
                
                # Create experiment name and directory
                EXP_NAME="${ENCODER_NAME}_${IMAGE_SIZE}_${SIDE_VIEW_STR}_${DECODER_NAME}"
                EXP_DIR="${MAIN_LOG_DIR}/${EXP_NAME}"
                mkdir -p $EXP_DIR
                
                echo "Running experiment: $EXP_NAME"
                echo "  Image Size: $IMAGE_SIZE"
                echo "  Side View: $SIDE_VIEW"
                echo "  Encoder: $ENCODER"
                echo "  Decoder: $DECODER"
                
                # Run the experiment and save output to log.txt
                (python src/train_hf.py \
                    --image_size $IMAGE_SIZE \
                    --side_view $SIDE_VIEW \
                    --preload False \
                    --encoder_checkpoint $ENCODER \
                    --decoder_checkpoint $DECODER \
                    --dataset_path $DATASET_PATH \
                    --today_date_str $TODAY_DATE \
                    --epoch $EPOCH \
                    --exp_name $EXP_NAME 2>&1 | tee "${EXP_DIR}/log.txt")        
                
                echo "Experiment completed: $EXP_NAME"
                echo "Logs saved to: ${EXP_DIR}/log.txt"
                echo "Benchmark saved to: ${EXP_DIR}/benchmark.txt"
                echo "-----------------------------------"
            done
        done
    done
done

echo "All experiments completed!"
echo "Results saved to: $MAIN_LOG_DIR"

# Make sure to make the script executable with:
# chmod +x run_experiments.sh
