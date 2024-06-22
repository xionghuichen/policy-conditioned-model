import argparse
import os
import random
import gym
import d4rl
import h5py

import numpy as np
import torch
import tensorflow as tf

import sys
sys.path.append('.')

from models.policy_conditioned_dynamics_model import PolicyEncoder, PolicyDecoder, PolicyConditionedDynamicsModel
from dynamics.pcm_dynamics import PCMDynamics
from models.dope_policy import D4RLPolicy, DMCPolicy
from models.components import TanhDiagGaussian
from utils.scaler import StandardScaler
from utils.termination_fns import get_termination_fn
from utils.data import SequenceDataset
from utils.logger import Logger, make_log_dirs, load_args
# from utils.dm_control_suite import ControlSuite
from utils.dmc_env_wrapper import DMCEnvWrapper, get_obs_map_fn
from train_scripts.env_dict_map import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="pcm")
    parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--load-exp-path", type=str, default="log/walker2d-medium-replay-v2/pcm&deterministic=True&grad_step=50&embedding_dim=128&policy_recon_weight=0.01&info=stochastic_recon_loss")
    parser.add_argument("--load-eval-policy-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")

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


def init(args):
    # init env and eval_policy_set
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
        eval_policy_set = [D4RLPolicy(args.load_eval_policy_path + f"_{idx}.pkl") for idx in range(11)]

    # init dynamics
    dynamics_model = PolicyConditionedDynamicsModel(
        input_dim=int(np.prod(args.obs_shape)+args.action_dim),
        output_dim=int(np.prod(args.obs_shape)+1),
        embedding_dim=args.embedding_dim,
        hidden_dims=args.dynamics_hidden_dims,
        device=args.device
    )

    policy_encoder = PolicyEncoder(
        input_dim=int(np.prod(args.obs_shape)+args.action_dim),
        hidden_dim=args.embedding_dim,
        weight_decay=args.weight_decay,
        device=args.device)

    dist = TanhDiagGaussian(
        latent_dim=200, 
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    
    policy_decoder = PolicyDecoder(
        input_dim=int(np.prod(args.obs_shape)),
        embedding_dim=args.embedding_dim,
        dist=dist,
        device=args.device
    )
    
    dynamics_optim = torch.optim.Adam(dynamics_model.parameters(), lr=args.dynamics_lr)
    encoder_optim = torch.optim.Adam(policy_encoder.parameters(), lr=args.dynamics_lr)
    decoder_optim = torch.optim.Adam(policy_decoder.parameters(), lr=args.dynamics_lr)
    scaler = StandardScaler(None, None)
    termination_fn = get_termination_fn(task=args.task.lower())

    dynamics = PCMDynamics(
        dynamics_model,
        policy_encoder,
        policy_decoder,
        dynamics_optim,
        encoder_optim,
        decoder_optim,
        scaler,
        termination_fn,
        clip_val=args.clip_val,
        grad_step=args.grad_step,
        policy_recon_weight=args.policy_recon_weight,
        deterministic=args.deterministic
    )

    return env, eval_policy_set, dynamics


def main(eval_args=get_args()):
    seeds_path = os.listdir(eval_args.load_exp_path)
    real_values = None
    for seed_path in seeds_path:
        if "seed" not in seed_path: continue
        exp_path = os.path.join(eval_args.load_exp_path, seed_path)
        args = load_args(os.path.join(exp_path, "record/hyper_param.json"))
        args.device = eval_args.device

        # seed
        random.seed(eval_args.seed)
        np.random.seed(eval_args.seed)
        torch.manual_seed(eval_args.seed)
        torch.cuda.manual_seed_all(eval_args.seed)
        torch.backends.cudnn.deterministic = True
        # env.seed(args.seed)

        # init
        env, eval_policy_set, dynamics = init(args)
        dynamics.load_state_dict(torch.load(os.path.join(exp_path, "model/dynamics.pth"), map_location=eval_args.device))
        dynamics.eval()

        print(seed_path)
        dynamics.set_for_eval_value_gap(env, eval_policy_set)

        if real_values is None:
            real_values = []
            for idx, policy in enumerate(eval_policy_set):
                print(f"eval_idx {idx}")
                real_value = dynamics.compute_real_value(policy, num_eval_episodes=eval_args.num_eval_episodes)
                real_values.append(real_value)

        fake_values = []
        for idx, policy in enumerate(eval_policy_set):
            print(f"eval_idx {idx}")
            fake_value = dynamics.eval_value_gap(policy, num_eval_episodes=eval_args.num_eval_episodes)
            fake_values.append(fake_value)

        real_values, fake_values = np.array(real_values), np.array(fake_values)
        value_min, value_max = real_values.min(), real_values.max()
        norm_real_values = (real_values - value_min) / (value_max - value_min)
        norm_fake_values = (fake_values - value_min) / (value_max - value_min)
        print(norm_real_values)
        print(norm_fake_values)
        absolute_error = (np.abs(norm_real_values - norm_fake_values)).mean()
        rank_correlation = np.corrcoef(norm_real_values, norm_fake_values)[0, 1]

        top_idxs = np.argsort(norm_fake_values)[-1:]
        regret = norm_real_values.max() - norm_real_values[top_idxs].max()
        print(f"absolute error: {absolute_error}")
        print(f"rank correlation: {rank_correlation}")
        print(f"regret: {regret}")


if __name__ == "__main__":
    main()