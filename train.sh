data_path="capstor/scratch/cscs/ljiayi/data/truck"
model_path="~/gaussian-splatting-implementations/speedy-splat_forked/output/wandb_truck"

python ~/gaussian-splatting-implementations/speedy-splat_forked/train.py \
  -s ${data_path} \
  -m ${model_path} \
  --eval \
  --iterations 30000