declare ckpt_paths=('')
declare datasets=('atlas-dermato' 'atlas-clinical' 'isic20' 'pad-ufes-20')

for path in ${ckpt_paths[@]}; do
    echo "Running model from path: $path ..."
    for ds in ${datasets[@]}; do
        
        python3 test_external_datasets.py --dataset $ds --ckpt_path $path
        
#         remember to add '--fromcl' as extra parameter if the checkpoint from the tested model went through contrastive pre-training
#         python3 test_external_datasets.py --dataset $ds --ckpt_path $path --fromcl

    done
done