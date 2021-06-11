LRS=(0.1 0.05 0.005 0.009 0.0001)
BALANCED=(0 1)
SCHEDULERS=('plateau' 'cosine')
BZ=(32 128 512)
method='baseline'

for LR in ${LRS[@]}; do
    for schedule in ${SCHEDULERS[@]}; do
        for bz in ${BZ[@]}; do
            for balanced in ${BALANCED[@]}; do
                echo " Running with Params: 
                        - LR: $LR
                        - Scheduler: $schedule
                        - Batch Size: $bz
                        - Balanced: $balanced"
                
                if [ $balanced = 1 ]; then
                    python3 finetuning_ssl.py --method $method --opt $schedule --batch_size $bz --lr $LR --balanced
                else
                    python3 finetuning_ssl.py --method $method --opt $schedule --batch_size $bz --lr $LR
                fi
            done
        done
    done
    
done