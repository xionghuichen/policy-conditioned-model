import argparse
import os
import random
import gym
import d4rl
import h5py

import numpy as np
import torch

from models.policy_conditioned_dynamics_model import PolicyEncoder, PolicyDecoder, PolicyConditionedDynamicsModel
from models.components import TanhDiagGaussian
from dynamics.pcm_dynamics import PCMDynamics
from mpc.cem import CEMOptimizer
from utils.scaler import StandardScaler
from utils.termination_fns import get_termination_fn
from utils.data import SequenceDataset
from utils.logger import Logger, make_log_dirs, load_args
from train_scripts.env_dict_map import *


LOAD_DYNAMICS_PATH = {
    # "halfcheetah-medium-replay-v2": "log/halfcheetah-medium-replay-v2/pam&deterministic=True&grad_step=50&embedding_dim=16&policy_recon_weight=0.01&info=stochastic_recon_loss/seed_1&timestamp_23-0223-101545",
     "halfcheetah-medium-replay-v2": "/home/ubuntu/chenruifeng/policy_conditioned_model/log/halfcheetah-medium-replay-v2/pcm&deterministic=True&grad_step=50&embedding_dim=128&policy_recon_weight=0.01&info=stochastic_recon_loss/seed_0&timestamp_24-0620-145156",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="pcm-cem-test")
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    
    parser.add_argument("--planning-horizon", type=int, default=30)
    parser.add_argument("--num-candidates", type=int, default=500)
    parser.add_argument("--cem-iters", type=int, default=25)
    parser.add_argument("--num-elites", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--info", type=str, default=None)

    return parser.parse_args()


def init(args):
    # create dynamics
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
        output_dim=int(args.action_dim),
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

    return dynamics


class MPCAgent:
    def __init__(self, dynamics, args):
        self.dynamics = dynamics
        self.action_dim = int(args.action_dim)

        # init cem optimizer
        ac_lb = np.tile(-np.ones(self.action_dim)*args.max_action, [args.planning_horizon])
        ac_ub = np.tile(np.ones(self.action_dim)*args.max_action, [args.planning_horizon])
        self.cem_optimizer = CEMOptimizer(
            planning_horizon=args.planning_horizon,
            action_dim=self.action_dim,
            max_iters=args.cem_iters,
            popsize=args.num_candidates,
            num_elites=args.num_elites,
            cost_function=self.cost_function,
            upper_bound=ac_ub,
            lower_bound=ac_lb,
            alpha=args.alpha,
            num_particals=1
        )

    @ torch.no_grad()
    def cost_function(self, current_obs, action_seq, num_particals):
        # action_sequences shape [num candidates, seq_len, action_dim]
        num_candidates, seq_len = action_seq.shape[0], action_seq.shape[1]
        returns = np.zeros(num_particals*num_candidates)
        
        # obss: [num_particals*num_candidates, obs_dim]
        obss = np.tile(current_obs, [num_particals*num_candidates, 1])
        # action_seq: [num_particals*num_candidates, seq_len, action_dim]
        action_seq = np.tile(action_seq, [num_particals, 1, 1])

        last_obs_acts = self.last_obs_act
        # last_obs_acts: [num_particals*num_candidates, obs_act_dim]
        last_obs_acts = np.tile(last_obs_acts, [num_particals*num_candidates, 1])
        if self.h_state is not None:
            h_states = self.h_state.repeat(1, num_particals*num_candidates, 1)
        else:
            h_states = self.h_state

        for t in range(seq_len):
            next_obss, rewards, _, info = self.dynamics.step(obss, action_seq[:, t], last_obs_acts, h_states)
            returns += rewards.flatten()
            last_obs_acts = np.concatenate([obss, action_seq[:, t]], -1)
            h_states = info["h_state"]
            obss = next_obss.copy()
        
        returns = returns.reshape(num_particals, num_candidates).mean(0)
        return -returns
    
    @ torch.no_grad()
    def select_action(self, obs, last_obs_act, h_state):
        self.last_obs_act = last_obs_act
        self.h_state = h_state

        init_sol = (self.cem_optimizer.ub + self.cem_optimizer.lb) / 2
        init_var = np.square(self.cem_optimizer.ub - self.cem_optimizer.lb) / 16
        sol = self.cem_optimizer.obtain_solution(init_sol, init_var, obs)
        action = sol[:self.action_dim]

        # update h_state
        last_obs_act = np.concatenate([obs, action.reshape(1, -1)], -1)
        _, h_state = self.dynamics.get_embedding(last_obs_act, h_state)
        return action, h_state


def eval_policy(agent, eval_env, logger):
    obs_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    obs = eval_env.reset()
    eval_ep_info_buffer = []
    num_episodes = 0
    episode_reward, episode_length = 0, 0
    last_obs_act = np.zeros((1, obs_dim+action_dim), dtype=np.float32)
    h_state = None

    while num_episodes < 1:
        action, h_state = agent.select_action(obs.reshape(1, -1), last_obs_act, h_state)
        next_obs, reward, terminal, _ = eval_env.step(action.flatten())

        episode_reward += reward
        episode_length += 1
        logger.log(f"step {episode_length}, episode reward: {episode_reward}")
        
        last_obs_act = np.concatenate([obs, action], axis=-1).reshape(1, -1)

        obs = next_obs.copy()

        if terminal or (episode_length >= 200):
            logger.log(f"ep_len: {episode_length}, ep_rew: {episode_reward}")
            eval_ep_info_buffer.append(
                {"episode_reward": episode_reward, "episode_length": episode_length}
            )
            num_episodes +=1
            episode_reward, episode_length = 0, 0
            obs = eval_env.reset()
            last_obs_act = np.zeros((1, obs_dim+action_dim), dtype=np.float32)
            h_state = None
    
    return {
        "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
        "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
    }


def main(eval_args=get_args()):
    args = load_args(os.path.join(LOAD_DYNAMICS_PATH[eval_args.task], "record/hyper_param.json"))
    args.device = eval_args.device
    eval_args.max_action = args.max_action
    eval_args.action_dim = args.action_dim
    env = gym.make(eval_args.task)

    # seed
    random.seed(eval_args.seed)
    np.random.seed(eval_args.seed)
    torch.manual_seed(eval_args.seed)
    torch.cuda.manual_seed_all(eval_args.seed)
    torch.backends.cudnn.deterministic = True
    # env.seed(args.seed)

    dynamics = init(args)
    dynamics.load_state_dict(torch.load(
        os.path.join(LOAD_DYNAMICS_PATH[eval_args.task], "model/dynamics.pth"),
        map_location=torch.device(eval_args.device)
    ))
    dynamics.eval()

    agent = MPCAgent(dynamics, eval_args)

    # log
    log_dirs = make_log_dirs(
        eval_args.task,
        eval_args.algo_name, eval_args.seed, vars(eval_args), record_params=["planning_horizon", "info"]
    )
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    eval_info = eval_policy(agent, env, logger)
    ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
    ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
    norm_ep_rew_mean = env.get_normalized_score(ep_reward_mean) * 100
    norm_ep_rew_std = env.get_normalized_score(ep_reward_std) * 100
    logger.log(f"normalized_episode_reward: {norm_ep_rew_mean}")
    logger.log(f"normalized_episode_reward_std: {norm_ep_rew_std}")


if __name__ == "__main__":
    main()