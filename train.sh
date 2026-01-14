data_path="/users/ljiayi/data/truck"
model_path="/users/ljiayi/speedy-splat_forked/output/wandb_truck_new_pruning"

python /users/ljiayi/speedy-splat_forked/train.py \
  -s ${data_path} \
  -m ${model_path} \
  --eval \
  --iterations 30000