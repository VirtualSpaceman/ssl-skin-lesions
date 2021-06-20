#!/bin/bash
declare temperatures=(0.1 0.5 1.0)
declare methods=("SimCLR" "SupCon")
declare lr_contrastive=(0.001)
declare batch_sizes=(80 512)
declare img_size=(224)

SECONDS=0
for temp in ${temperatures[@]}; do
    for method in ${methods[@]}; do
        for batch in ${batch_sizes[@]}; do
            for lr_contrast in ${lr_contrastive[@]}; do
                for size in ${img_size[@]}; do
                    # Reset BASH time counter
                    SECONDS=0
                    printf "Running Experiment with following parameters: 
                        - Temperature: ${temp}
                        - LR Contrastive: ${lr_contrast}
                        - Method: ${method}
                        - Batch Size: ${batch}
                        - Image Size: ${size}
                        - Unbalanced DataLoader
                        "
                    python3 main_isic_supcon.py --pretrained --cosine \
                                                      --size $size --batch_size $batch \
                                                      --learning_rate $lr_contrast \
                                                      --epochs 200  \
                                                      --temp $temp --method $method > \
                                                        ./logs/log_train_unbalanced_pretrained_${method}_lr_${lr_contrast}_temp_${temp}_bsz_${batch}_img_${size}.txt
                    ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
                    echo "Time $ELAPSED"

                done
            done
        done
    done
done
