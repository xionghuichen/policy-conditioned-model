import argparse
import os
import random
import gym
import d4rl
import h5py

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.policy import MOPOPolicy

from models.vanilla_dynamics_model import VanillaDynamicsModel
from dynamics.vanilla_dynamics import VanillaDynamics
from models.components import TanhDiagGaussian as TanhDiagGaussian1
from utils.scaler import StandardScaler as StandardScaler1
from utils.termination_fns import get_termination_fn as get_termination_fn1
from utils.logger import Logger, make_log_dirs, load_args
from train_scripts.env_dict_map import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="vanilla-dynamics")
    parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--load-exp-path", type=str, default="log/walker2d-medium-replay-v2/vanilla-dynamics&deterministic=True&info=None/seed_1&timestamp_23-0227-083303")
    parser.add_argument("--load-eval-policy-path", type=str, default="log/walker2d-medium-replay-v2/mopo&penalty_coef=2.5&rollout_length=1/seed_0&timestamp_23-0318-144554/checkpoint")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def load_eval_policy(load_path, device):
    args_path = os.path.dirname(os.path.dirname(load_path))
    args_path = os.path.join(args_path, "record/hyper_param.json")
    args = load_args(args_path)

    # create policy model
    actor_backbone = MLP(input_dim=int(np.prod(args.obs_shape)), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=int(np.prod(args.obs_shape) + args.action_dim), hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=int(np.prod(args.obs_shape) + args.action_dim), hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=int(args.action_dim),
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, device)
    critic1 = Critic(critic1_backbone, device)
    critic2 = Critic(critic2_backbone, device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    alpha = (args.target_entropy, log_alpha, alpha_optim)

    # create policy
    policy = MOPOPolicy(
        None,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha
    )
    policy.load_state_dict(torch.load(load_path, map_location=device))

    return policy


def init(args):
    # init env and eval_policy_set
    env = gym.make(args.task)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]
    load_policy_paths = os.listdir(args.load_eval_policy_path)
    eval_policy_set = [load_eval_policy(os.path.join(args.load_eval_policy_path, path), args.device) for path in load_policy_paths]

    # init dynamics
    dynamics_model = VanillaDynamicsModel(
        input_dim=np.prod(args.obs_shape)+args.action_dim,
        output_dim=np.prod(args.obs_shape)+1,
        hidden_dims=args.dynamics_hidden_dims,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(dynamics_model.parameters(), lr=args.dynamics_lr)
    scaler = StandardScaler1(None, None)
    termination_fn = get_termination_fn1(task=args.task.lower())

    dynamics = VanillaDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        deterministic=args.deterministic
    )

    return env, eval_policy_set, dynamics


def main(eval_args=get_args()):
    args = load_args(os.path.join(eval_args.load_exp_path, "record/hyper_param.json"))
    args.device = eval_args.device
    args.load_eval_policy_path = eval_args.load_eval_policy_path

    # seed
    random.seed(eval_args.seed)
    np.random.seed(eval_args.seed)
    torch.manual_seed(eval_args.seed)
    torch.cuda.manual_seed_all(eval_args.seed)
    torch.backends.cudnn.deterministic = True
    # env.seed(args.seed)

    # init
    env, eval_policy_set, dynamics = init(args)
    dynamics.load_state_dict(torch.load(os.path.join(eval_args.load_exp_path, "model/dynamics.pth"), map_location=eval_args.device))
    dynamics.eval()

    dynamics.set_for_eval_value_gap(env, eval_policy_set)

    real_values = []
    for idx, policy in enumerate(eval_policy_set):
        print(f"eval_idx {idx}")
        real_value = dynamics.compute_real_value(policy, num_eval_episodes=eval_args.num_eval_episodes, gamma=1)
        real_value = env.get_normalized_score(real_value) * 100
        real_values.append(real_value)

    fake_values = []
    for idx, policy in enumerate(eval_policy_set):
        print(f"eval_idx {idx}")
        fake_value = dynamics.eval_value_gap(policy, num_eval_episodes=eval_args.num_eval_episodes, gamma=1)
        fake_value = env.get_normalized_score(fake_value) * 100
        fake_values.append(fake_value)

    real_values, fake_values = np.array(real_values), np.array(fake_values)
    print(real_values)
    print(fake_values)
    print(real_values.mean(), real_values.std())
    selected_idx = np.argsort(fake_values)[-1:]
    print(real_values[selected_idx])


if __name__ == "__main__":
    main()