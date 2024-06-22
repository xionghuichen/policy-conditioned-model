nohup python train_scripts/train_vanilla_dynamics.py \
    --task "halfcheetah-medium-replay-v2" \
    --seed 0 \
    --device "cuda:0" > halfcheetah_mlp.txt &

nohup python train_scripts/train_vanilla_dynamics.py \
    --task "hopper-medium-replay-v2" \
    --seed 0 \
    --device "cuda:0" > hopper_mlp.txt &

nohup python train_scripts/train_vanilla_dynamics.py \
    --task "walker2d-medium-replay-v2" \
    --seed 0 \
    --device "cuda:0" > walker2d_mlp.txt &