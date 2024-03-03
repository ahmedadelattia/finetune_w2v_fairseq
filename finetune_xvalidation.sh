#runs x-validation using finetune.sh

# Path: finetune.sh
w2v_name=$1
resume=$2

#remove .pt from the w2v_name if it exists
w2v_name=${w2v_name%.pt}

for fold in 144 622 2619 2709 2944 4724; do
    echo "Running on fold: $fold"
    bash finetune.sh $w2v_name $fold $resume
done
