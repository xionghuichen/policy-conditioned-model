import argparse
import os
import random
import gym
import d4rl.gym_mujoco
import h5py

import numpy as np
import torch
import tensorflow as tf

from models.vanilla_dynamics_model import VanillaDynamicsModel
from dynamics.vanilla_dynamics import VanillaDynamics
from models.dope_policy import D4RLPolicy, DMCPolicy
from utils.scaler import StandardScaler
from utils.termination_fns import get_termination_fn
from utils.data import BaseDataset
from utils.logger import Logger, make_log_dirs
# from utils.dm_control_suite import ControlSuite
from utils.dmc_env_wrapper import DMCEnvWrapper, get_obs_map_fn
from train_scripts.env_dict_map import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mlp-dynamics")
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    parser.add_argument("--load-dataset-path", type=str, default=None)
    parser.add_argument("--load-eval-policy-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--dynamics-lr", type=float, default=1e-4)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--deterministic", type=bool, default=True)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, default=201)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--info", type=str, default=None)

    return parser.parse_args()


def set_tf_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def get_eval_policy_path(root_dir):
    paths = []
    for algo_id in os.listdir(root_dir):
        for policy_id in os.listdir(os.path.join(root_dir, algo_id)):
            paths.append(os.path.join(root_dir, algo_id, policy_id))
    return paths


def main(args=get_args()):
    assert args.task in data_map
    kwargs = vars(args)
    kwargs.update(data_map[args.task])
    args = argparse.Namespace(**kwargs)

    # create env and dataset
    if args.dataset_type == "rlunplugged":
        # set_tf_gpu()
        # task = ControlSuite(args.task)
        # env = DMCEnvWrapper(task.environment, keys=task._keys)
        # obs_map_fn = get_obs_map_fn(args.task)
        # eval_policy_set = [
        #     DMCPolicy(tf.saved_model.load(os.path.join(policy_path)), obs_map_fn) \
        #     for policy_path in get_eval_policy_path(args.load_eval_policy_path)
        # ]
        # args.obs_shape = task.obs_shape
        # args.action_dim = task.action_dim
        args.max_action = 1.0
    else:
        env = gym.make(args.task)
        args.obs_shape = env.observation_space.shape
        args.action_dim = np.prod(env.action_space.shape)
        args.max_action = env.action_space.high[0]
        eval_policy_set = [D4RLPolicy(args.load_eval_policy_path + f"_{idx}.pkl") for idx in range(10)]

    if args.load_dataset_path is None:
        dataset = env.get_dataset()
        dataset["rewards"] = dataset["rewards"].reshape(-1, 1)
        dataset = BaseDataset(dataset, device=args.device)
    else:
        dataset = {}
        with h5py.File(args.load_dataset_path, 'r') as f:
            for k in f.keys():
                dataset[k] = f[k][:]
            dataset = BaseDataset(dataset, device=args.device)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    # env.seed(args.seed)

    # create dynamics
    dynamics_model = VanillaDynamicsModel(
        input_dim=np.prod(args.obs_shape)+args.action_dim,
        output_dim=np.prod(args.obs_shape)+1,
        hidden_dims=args.dynamics_hidden_dims,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(dynamics_model.parameters(), lr=args.dynamics_lr)
    scaler = StandardScaler(dataset.obs_mean, dataset.obs_std)
    termination_fn = get_termination_fn(task=args.task.lower())

    dynamics = VanillaDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        deterministic=args.deterministic
    )

    # log
    task_name = os.path.splitext(os.path.basename(args.load_dataset_path))[0] if args.load_dataset_path is not None else args.task
    log_dirs = make_log_dirs(
        task_name,
        args.algo_name, args.seed, vars(args), record_params=["deterministic", "info"]
    )
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    dynamics.set_for_eval_value_gap(env, eval_policy_set)
    dynamics.train_dynamics(dataset, args.batch_size, logger, max_epoch=args.max_epoch, eval_freq=10)


if __name__ == "__main__":
    main()