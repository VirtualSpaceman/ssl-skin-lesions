#!/bin/bash
declare split_folders=('/datasplits/isic2019/splits_1_percent_1/' '/datasplits/isic2019/splits_1_percent_2/' '/datasplits/isic2019/splits_1_percent_3/' '/datasplits/isic2019/splits_1_percent_4/' '/datasplits/isic2019/splits_1_percent_5/')
declare method='SupCon'
declare temps=(1.0 0.5 0.5)
declare batch=32
declare img_size=224
declare lr_contrast=0.001
declare balanced=(1 1 0)

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
                                             --epochs 200 --cosine --size $img_size
        else
        python3 main_isic_supcon.py --pretrained --size $img_size --temp ${temps[$i]} --print_freq 5 \
                                             --method $method --learning_rate $lr_contrast \
                                             --batch_size $batch --split_folder $split_folder \
                                             --epochs 200 --cosine --size $img_size
        fi 
        ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
        echo "Time $ELAPSED"
    done

done