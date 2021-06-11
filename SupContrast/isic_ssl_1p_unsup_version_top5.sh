#!/bin/bash
declare split_folders=('/experimentos/isic2019/new_splits/splits_1_percent_1/' '/experimentos/isic2019/new_splits/splits_1_percent_2/' '/experimentos/isic2019/new_splits/splits_1_percent_3/' '/experimentos/isic2019/new_splits/splits_1_percent_4/' '/experimentos/isic2019/new_splits/splits_1_percent_5/')
declare method='SimCLR'
declare temps=(0.1 0.1 0.5 1.0)
declare batch=32
declare img_size=224
declare lr_contrast=0.001
declare balanced=(1 0 0 0)
declare epochs=(200 50 200 200)

# get length of an array
declare tam_array=${#temps[@]}

# use for loop to read all values and indexes
for (( i=0; i<${tam_array}; i++ )); do
    for split_folder in ${split_folders[@]}; do
        SECONDS=0
        printf "Running Experiment with following parameters: 
        - Temperature: ${temps[$i]}
        - LR Contrastive: ${lr_contrast}
        - Method: ${method}
        - Batch Size: ${batch}
        - Image Size: ${size}
        - Balanced DataLoader: ${balanced[$i]}
        - Split folder: ${split_folder}
        "
        if [ ${balanced[$i]} = 1 ]; then
        python3 main_isic_supcon.py --pretrained --size $img_size --temp ${temps[$i]} --print_freq 5 \
                                             --method $method --learning_rate $lr_contrast --balanced \
                                             --batch_size $batch --split_folder $split_folder \
                                             --epochs ${epochs[$i]} --cosine --size $img_size
        else
        python3 main_isic_supcon.py --pretrained --size $img_size --temp ${temps[$i]} --print_freq 5 \
                                             --method $method --learning_rate $lr_contrast \
                                             --batch_size $batch --split_folder $split_folder \
                                             --epochs ${epochs[$i]} --cosine --size $img_size
        fi 
        ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
        echo "Time $ELAPSED"
    done
done