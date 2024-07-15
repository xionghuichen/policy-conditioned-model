<h1 align="center">Policy-conditioned Environment Models are More Generalizable</h1>

The official implementation of [*Policy-conditioned Environment Models are More Generalizable*](https://openreview.net/pdf?id=g9mYBdooPA). 

Please view [Our Project Page](https://policy-conditioned-model.github.io/) for more details.


## Installation

Download and install the main code from [Policy Conditioned Model](https://github.com/xionghuichen/policy-conditioned-model).

```
git clone https://github.com/xionghuichen/policy-conditioned-model.git
cd policy-conditioned-model
pip install -e .
```

Install the [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit).

```
cd ..
git clone https://github.com/yihaosun1124/OfflineRL-Kit.git
cd OfflineRL-Kit
pip install -e .
```

## Usage
Train a policy-conditioned model:
```
python train_scripts/train_pcm.py
```
Evaluate policies within the policy-conditioned model:
```
python eval_scripts/eval_pcm.py
```
Perform offline policy selection based on the policy-conditioned model:
```
python eval_scripts/offline_policy_selection.py
```
Perform model predictive control using the policy-conditioned model:
```
python mpc/mpc_cem_by_pcm.py
```

## Citation

If you use this implementation in your work, please cite us with the following:

```
@inproceedings{
Policy-Conditioned Model,
title={Policy-conditioned Environment Models are More Generalizable},
author={Ruifeng Chen and Xiong-Hui Chen and Yihao Sun and Siyuan Xiao and Minhui Li and Yang Yu},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=g9mYBdooPA}
}
```
