#w2v_name is the first argument
export HYDRA_FULL_ERROR=1
w2v_name=$1
dataset=$2
fold=$3
resume=$4

outdir=./model_outputs/
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
  model_path=$(realpath $model_path)
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
mkdir -p $outdir_fold
cd $outdir_fold
echo "Current directory: $(pwd)"
echo "outdir_fold: $outdir_fold"
echo "dataset: $dataset"
echo "Running on fold: $fold"

save_dir=$fold

manifest_path=$curr_dir/manifest/$dataset

args="task.data="$curr_dir/manifest/$fold""
args+="model.w2v_path='$model_path' "
if [[ $resume == "true" ]] ; then
  restore_file=$(ls -d $outdir_fold/$fold/*/*/* | tail -n 1)/checkpoint_last.pt
  #assert that the file exists
  if [ ! -f $restore_file ]; then
    echo "File $restore_file does not exist"
    exit 1
  fi

  args+="checkpoint.restore_file=""$restore_file" "
fi

args+="common.wandb_project="$wandb_project""
args+="checkpoint.save_dir="./$fold" "
args+="distributed_training.distributed_world_size=1 "
args+="--config-dir $curr_dir/config/ "
args+="--config-name $config_name"


echo args: $args
fairseq-hydra-train $args