#!/bin/bash
#w2v_name is the first argument
export HYDRA_FULL_ERROR=1
w2v_name=$1
dataset=$2
fold=$3
restore_file=${4:-""}
resume=${5:-false}
lr=$6
dev=$7
echo dev: $dev


if $resume == "true" && [[ -n $restore_file ]]; then
  echo "Restore file restores from a given file. Resume resumes from the last checkpoint. Please provide a restore file or set resume to false"
  exit 1
fi

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


if [[ -n $restore_file ]]; then
  echo "Restoring from file: $restore_file"
  #if restore_file ends with checkpoint_last.pt, warn the user to use checkpoint_best.pt instead
  if [[ $restore_file == *checkpoint_last.pt ]]; then
    echo "Warning: You are restoring from checkpoint_last.pt. It is recommended to use checkpoint_best.pt instead"
  fi
  #assert that the file exists
  if [[ ! -f "$restore_file" ]]; then
    echo "File $restore_file does not exist"
    exit 1
  fi

elif [[ $resume == "true" ]] ; then
  #example ./model_outputs/xlsr2_300m/2944/outputs/2024-03-03/14-36-20/2944/checkpoint_last.pt
  restore_file=$(ls -t outputs/*/*/"$fold"/checkpoint_last.pt)
  #if multiple files are found, take the latest one alphabetically
  if [[ $(echo "$restore_file" | wc -l) -gt 1 ]]; then
    echo "Multiple files found, taking the latest one"
    echo "$restore_file"
    restore_file=$(echo "$restore_file" | head -n 1)
  fi
  restore_file=$(realpath "$restore_file")
  echo "restore_file: "$restore_file""

  #assert that the file exists
  if [[ ! -f "$restore_file" ]]; then
    echo "File $restore_file does not exist"
    exit 1
  fi


else
  restore_file=""
fi

if [[ $dev == "true" ]]; then
    wandb_project=""
fi

#if lr is not provided, use the default value
if [[ -z $lr ]]; then
  lr=0.0003
fi

echo "fairseq-hydra-train \
  model.w2v_path="$model_path" \
  task.data="$manifest_path/$fold" \
  checkpoint.save_dir="$save_dir" \
  checkpoint.restore_file="$restore_file" \
  common.wandb_project="$wandb_project" \
  optimization.lr=[$lr] \
  distributed_training.distributed_world_size=1 \
  --config-dir $curr_dir/config/ \
  --config-name $config_name \
  "
  
if $resume == "false" && [[ -n $restore_file ]]; then
echo "Transfer learning from file: $restore_file"
echo "Warning: You are transfer learning from a fully trained model so the optimizer, lr_scheduler, and dataloader are reset"
echo "If you want to resume training from the last trained checkpoint with the same configuration, set resume to true and do not provide a restore file"
  fairseq-hydra-train \
    model.w2v_path="$model_path" \
    task.data="$manifest_path/$fold" \
    checkpoint.save_dir="$save_dir" \
    checkpoint.restore_file="$restore_file" \
    checkpoint.reset_optimizer=true \
    checkpoint.reset_lr_scheduler=true \
    checkpoint.reset_dataloader=true \
    common.wandb_project="$wandb_project" \
    optimization.lr=[$lr] \
    distributed_training.distributed_world_size=1 \
    --config-dir $curr_dir/config/ \
    --config-name $config_name \
    --restore

elif [[ -n restore_file && resume == "true" ]]; then
  fairseq-hydra-train \
    model.w2v_path="$model_path" \
    task.data="$manifest_path/$fold" \
    checkpoint.save_dir="$save_dir" \
    checkpoint.restore_file="$restore_file" \
    common.wandb_project="$wandb_project" \
    optimization.lr=[$lr] \
    distributed_training.distributed_world_size=1 \
    --config-dir $curr_dir/config/ \
    --config-name $config_name \
    --restore

else
  fairseq-hydra-train \
    model.w2v_path="$model_path" \
    task.data="$manifest_path/$fold" \
    checkpoint.save_dir="$save_dir" \
    checkpoint.restore_file="$restore_file" \
    common.wandb_project="$wandb_project" \
    optimization.lr=[$lr] \
    distributed_training.distributed_world_size=1 \
    --config-dir $curr_dir/config/ \
    --config-name $config_name 
fi