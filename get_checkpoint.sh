#machine defaults to jagupard35, but can be changed with -m flag
machine=jagupard37
while getopts m: flag
do
    case "${flag}" in
        m) machine=${OPTARG};;
    esac
done

#path is a required argument 
path=$1



#example usage: bash get_checkpoint.sh -p Projects/finetune_w2v_fairseq/checkpoints/finetune_w2v_fairseq_1178_20000 -m jagupard35

if [ -z "$path" ]
then
    echo "Path is a required argument"
    exit 1
fi



mkdir -p $path

echo "rsync -avz /$machine/scr/aadel4/Projects/finetune_w2v_fairseq/$path $path --exclude checkpoint_last.pt --exclude checkpoint_1178_20000.pt"
rsync -av --info=progress2 --ignore-existing --no-compress --whole-file --partial --inplace  aadel4@scdt.stanford.edu:/$machine/scr/aadel4/Projects/finetune_w2v_fairseq/$path $path  --exclude checkpoint_1178_20000.pt