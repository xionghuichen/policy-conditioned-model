nohup python train_scripts/train_pcm.py \
    --task "halfcheetah-medium-replay-v2" \
    --seed 0 \
    --device "cuda:0" > halfcheetah_pcm.txt &

nohup python train_scripts/train_pcm.py \
    --task "hopper-medium-replay-v2" \
    --seed 0 \
    --device "cuda:0" > hopper_pcm.txt &

nohup python train_scripts/train_pcm.py \
    --task "walker2d-medium-replay-v2" \
    --seed 0 \
    --device "cuda:0" > walker2d_pcm.txt &