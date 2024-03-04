#runs x-validation using finetune.sh

# Path: finetune.sh
w2v_name=$1
dataset=$2
resume=$3

#remove .pt from the w2v_name if it exists
w2v_name=${w2v_name%.pt}
if [[ $dataset == "NCTE" ]]; then
    #folds are defined in manifest/NCTE. extract them
    folds=$(ls manifest/NCTE | grep fold)
    echo "Folds: $folds"
    for fold in $folds; do
        echo "Running on fold: $fold"
        bash finetune.sh $w2v_name $dataset $fold $resume
    dones
fi