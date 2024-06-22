import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from copy import deepcopy


class VanillaDynamics(object):
    def __init__(self, model, optim, scaler, terminal_fn, deterministic=False):
        super().__init__()

        self.model = model
        self.optim = optim
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self.deterministic = deterministic

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    @ torch.no_grad()
    def step(self, obs, act):
        "imagine single forward step"
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        if len(act.shape) == 1:
            act = act.reshape(1, -1)
        
        raw_obs = obs.copy()
        obs = self.scaler.transform(obs)
        obs_act = np.concatenate([obs, act], axis=-1)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy().astype(np.float32)
        logvar = logvar.cpu().numpy().astype(np.float32)
        mean[..., :-1] += raw_obs

        if self.deterministic:
            next_obs = mean[..., :-1]
            reward = mean[..., -1:]
        else:
            std = np.sqrt(np.exp(logvar))
            samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)
            next_obs = samples[..., :-1]
            reward = samples[..., -1:]

        terminal = self.terminal_fn(raw_obs, act, next_obs)
        info = {}

        return next_obs, reward, terminal, info

    def train_dynamics(self, dataset, batch_size, logger, max_epoch=None, eval_freq=None, num_eval_episodes=10):
        data_size = len(dataset)
        holdout_size = min(data_size * 0.2, 1000)
        training_set, holdout_set = torch.utils.data.random_split(
            dataset, [data_size-holdout_size, holdout_size]
        )
        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        validate_loader = DataLoader(holdout_set, batch_size=batch_size, shuffle=False)
        
        epoch = 0
        cnt = 0
        holdout_loss = 1e10
        best_state_dict = None

        real_values = []
        for idx, policy in enumerate(self.eval_policy_set):
            print("eval pi idx", idx)
            real_value = self.compute_real_value(policy, num_eval_episodes=num_eval_episodes)
            real_values.append(real_value)

        while True:
            epoch += 1
            self.train()
            for obss, actions, _, delta_obs_rews in train_loader:
                train_loss = self.train_onestep(obss, actions, delta_obs_rews)
            new_holdout_loss = self.validate(validate_loader)
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", new_holdout_loss)

            if (eval_freq is not None) and ((epoch-1) % eval_freq == 0):
                fake_values = []
                for idx, policy in enumerate(self.eval_policy_set):
                    print("eval pi idx", idx)
                    fake_value = self.eval_value_gap(policy, num_eval_episodes=num_eval_episodes)
                    fake_values.append(fake_value)
                real_values, fake_values = np.array(real_values), np.array(fake_values)
                value_min, value_max = real_values.min(), real_values.max()
                norm_real_values = (real_values - value_min) / (value_max - value_min)
                norm_fake_values = (fake_values - value_min) / (value_max - value_min)
                absolute_error = (np.abs(norm_real_values - norm_fake_values)).mean()
                logger.logkv(f"eval/init_rollout_value_gap", absolute_error)
                for idx, vg in enumerate(np.abs(norm_real_values - norm_fake_values)):
                    logger.logkv(f"eval_pi/init_rollout_value_gap_idx-{idx}", vg)
                
                torch.save(self.state_dict(), os.path.join(logger.checkpoint_dir, f"dynamics_{epoch}.pth"))
            
                    
            logger.set_timestep(epoch)
            logger.dumpkvs()

            improvement = (holdout_loss - new_holdout_loss) / holdout_loss
            if improvement > 0.01:
                holdout_loss = new_holdout_loss
                best_state_dict = deepcopy(self.state_dict())
                cnt = 0
            else:
                cnt += 1
            if max_epoch is not None:
                if epoch >= max_epoch: break
            else:
                if cnt >= 5: break
        
        self.load_state_dict(best_state_dict)
        torch.save(self.state_dict(), os.path.join(logger.model_dir, "dynamics.pth"))

    def train_onestep(self, obss, actions, delta_obs_rews):
        obs_acts = torch.cat([obss, actions], -1)
        mean, logvar = self.model(obs_acts)

        if self.deterministic:
            loss = ((mean - delta_obs_rews) ** 2).mean()
        else:
            inv_var = torch.exp(-logvar)
            mse_loss_inv = (torch.pow(mean - delta_obs_rews, 2) * inv_var).mean()
            var_loss = logvar.mean()
            dynamics_loss = mse_loss_inv + var_loss
            dynamics_loss += (0.01 * self.dynamics_model.max_logvar.sum() - 0.01 * self.dynamics_model.min_logvar.sum())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    @ torch.no_grad()
    def validate(self, validate_loader):
        self.eval()
        losses = []
        for obss, actions, _, delta_obs_rews in validate_loader:
            obs_acts = torch.cat([obss, actions], -1)
            mean, logvar = self.model(obs_acts)
            loss = ((mean - delta_obs_rews) ** 2).mean()
            losses.append(loss.item())
        return np.mean(losses)
    
    def set_for_eval_value_gap(self, real_env, eval_policy_set):
        self.real_env = real_env
        self.real_env.reset()
        self.eval_policy_set = eval_policy_set

    def compute_real_value(self, policy, gamma=0.995, num_eval_episodes=10):
        # eval in real env
        real_ep_info = []
        obs = self.real_env.reset()
        num_episodes = 0
        value, episode_length = 0, 0

        while num_episodes < num_eval_episodes:
            action = policy.select_action(obs, deterministic=False)
            next_obs, reward, terminal, _ = self.real_env.step(action.flatten())
            value += reward * (gamma ** episode_length)
            episode_length += 1

            obs = next_obs.copy()

            if terminal:
                real_ep_info.append(
                    {"value": value, "episode_length": episode_length}
                )
                num_episodes += 1
                value, episode_length = 0, 0
                obs = self.real_env.reset()

        real_value = np.mean([info["value"] for info in real_ep_info])
        return real_value

    def eval_value_gap(self, policy, gamma=0.995, num_eval_episodes=10):
        
        # eval in fake env
        fake_ep_info = []
        obs = self.real_env.reset()
        obs_dim = obs.shape[-1]
        step = 0
        num_episodes = 0
        value, episode_length = 0, 0
        self.eval()

        while num_episodes < num_eval_episodes:
            obs = obs.reshape(-1, obs_dim)
            action = policy.select_action(obs, deterministic=False)
            next_obs, reward, terminal, _ = self.step(obs, action)
            reward, terminal = reward.flatten()[0], terminal.flatten()[0]
            value += reward * (gamma ** episode_length)
            episode_length += 1
            step += 1
            obs = next_obs.copy()
            if terminal or step >= 1000:
                fake_ep_info.append(
                    {"value": value, "episode_length": episode_length}
                )
                step = 0
                num_episodes += 1
                value, episode_length = 0, 0
                obs = self.real_env.reset()

        fake_value = np.mean([info["value"] for info in fake_ep_info])

        return fake_value

    def state_dict(self):
        return {
            "dynamics_model": self.model.state_dict(),
            "scaler/mu": torch.as_tensor(np.array(self.scaler.mu)),
            "scaler/std": torch.as_tensor(np.array(self.scaler.std))
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["dynamics_model"])
        self.scaler.mu = state_dict["scaler/mu"].cpu().numpy().flatten(),
        self.scaler.std = state_dict["scaler/std"].cpu().numpy().flatten()
