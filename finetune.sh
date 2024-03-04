#!/bin/bash
#w2v_name is the first argument
export HYDRA_FULL_ERROR=1
w2v_name=$1
dataset=$2
fold=$3
resume=$4
dev=$5
echo dev: $dev

outdir=./model_outputs/
if [[ $dev == "true" ]]; then
  outdir=./model_outputs_dev/
fi
curr_dir=$(pwd)
#remove .pt from the w2v_name if it exists
w2v_name=${w2v_name%.pt}
if [[ $w2v_name == *"/"* ]]; then
  model_path=$w2v_name.pt
  wandb_project=finetune_w2v2_continued_pretraining
  outdir+=continued_pretraining/
else
  model_path=$curr_dir/../pretrain_w2v_fairseq/base_models/$w2v_name.pt
  #get absolute path
  model_path=$pwd/$model_path
  echo "model_path: $model_path"

  wandb_project=finetune_w2v2_new_config_$w2v_name
  outdir+=$w2v_name

fi


config_name=large
echo w2v_name: $w2v_name
echo model_path: $model_path
echo wandb_project: $wandb_project
echo config_name: $config_name
#make directory for each fold
outdir_fold=${outdir}/$fold
mkdir -p "$outdir_fold"
cd "$outdir_fold"
echo "Current directory: $(pwd)"
echo "outdir_fold: $outdir_fold"
echo "dataset: $dataset"
echo "Running on fold: $fold"
save_dir=$fold

manifest_path=$curr_dir/manifest/$dataset
<<<<<<< HEAD
#replace spaces in $fold with escaped space
fold=${fold// /\\ }
args=task.data=$manifest_path/$fold 
args+=" model.w2v_path"=$model_path 
=======
>>>>>>> 8115ff5 (fixed space bug. fixed manifist)
if [[ $resume == "true" ]] ; then
  #example ./model_outputs/xlsr2_300m/2944/outputs/2024-03-03/14-36-20/2944/checkpoint_last.pt
  restore_file=$(ls -t outputs/*/*/"$fold"/checkpoint_last.pt)
  restore_file=$(realpath $restore_file)
  echo "restore_file: $restore_file"

  #assert that the file exists
  if [[ ! -f $restore_file ]]; then
    echo "File $restore_file does not exist"
    exit 1
  fi

<<<<<<< HEAD
  args+=checkpoint.restore_file=$restore_file 
fi

args+=" common.wandb_project"=$wandb_project 
args+=" checkpoint.save_dir"=./$fold 
args+=" distributed_training.distributed_world_size=1 "
args+="--config-dir $curr_dir/config/ "
args+="--config-name $config_name"
=======
else
  restore_file=""
fi

if [[ $dev == "true" ]]; then
    wandb_project=""
fi
>>>>>>> 8115ff5 (fixed space bug. fixed manifist)


fairseq-hydra-train \
  model.w2v_path="$model_path" \
  task.data="$manifest_path/$fold" \
  checkpoint.save_dir="$save_dir" \
  common.wandb_project="$wandb_project" \
  distributed_training.distributed_world_size=1 \
  --config-dir $curr_dir/config/ \
  --config-name $config_name \
  
