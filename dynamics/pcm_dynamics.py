import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm


class PCMDynamics(object):
    def __init__(
        self,
        dynamics_model,
        policy_encoder,
        policy_decoder,
        dynamics_model_optim,
        encoder_optim,
        decoder_optim,
        scaler,
        terminal_fn,
        clip_val,
        grad_step,
        policy_recon_weight,
        deterministic=True,
        stop_dynamics_grad_to_encoder=False,
    ):
        super().__init__()

        self.dynamics_model = dynamics_model
        self.policy_encoder = policy_encoder
        self.policy_decoder = policy_decoder

        self.dynamics_model_optim = dynamics_model_optim
        self.encoder_optim = encoder_optim
        self.decoder_optim = decoder_optim

        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self.clip_val = clip_val
        self.grad_step = grad_step
        self.policy_recon_weight = policy_recon_weight
        self.deterministic = deterministic

        self.stop_dynamics_grad_to_encoder = stop_dynamics_grad_to_encoder

    def train(self):
        self.dynamics_model.train()
        self.policy_encoder.train()
        self.policy_decoder.train()

    def eval(self):
        self.dynamics_model.eval()
        self.policy_encoder.eval()
        self.policy_decoder.eval()

    @ torch.no_grad()
    def get_embedding(self, last_obs_act, h_state):
        if last_obs_act.sum() != 0.:
            obs_dim = self.policy_decoder.input_dim
            last_obs, last_act = last_obs_act[:, :obs_dim].copy(), last_obs_act[:, obs_dim:].copy()
            # obs normalization
            last_obs = self.scaler.transform(last_obs)
            last_obs_act = np.concatenate([last_obs, last_act], axis=-1)
        last_obs_act = np.expand_dims(last_obs_act, 1)
        embedding, h_state = self.policy_encoder(last_obs_act, h_state)
        return embedding, h_state

    @ torch.no_grad()
    def step(self, obs, action, last_obs_act, h_state):
        # norm obs
        raw_obs = obs.copy()
        obs = self.scaler.transform(obs)
        obs_act = np.concatenate([obs, action], axis=-1)

        embedding, h_state = self.get_embedding(last_obs_act, h_state)
        mean, logvar = self.dynamics_model(obs_act, embedding)

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

        terminal = self.terminal_fn(raw_obs, action, next_obs)
        info = {'h_state': h_state}

        return next_obs, reward, terminal, info

    @ torch.no_grad()
    def step_with_embedding(self, obs, action, embedding):
        # norm obs
        raw_obs = obs.copy()
        obs = self.scaler.transform(obs)
        obs_act = np.concatenate([obs, action], axis=-1)

        # embedding, h_state = self.get_embedding(last_obs_act, h_state)
        mean, logvar = self.dynamics_model(obs_act, embedding)

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

        terminal = self.terminal_fn(raw_obs, action, next_obs)
        info = None

        return next_obs, reward, terminal, info

    def train_dynamics(self, dataset, batch_size, logger, max_epoch=None, eval_freq=10, num_eval_episodes=10):
        data_size = len(dataset)
        holdout_size = min(int(data_size * 0.2), 10)
        training_set, holdout_set = torch.utils.data.random_split(dataset, [data_size - holdout_size, holdout_size])
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
            
            train_losses = {}
            for obss, actions, prev_obs_acts, \
                delta_obs_rews, rtgs, stgs, masks in tqdm(train_loader):

                # truncate gradient for rnn
                train_loss, h_states = self.train_onestep(
                    obss[:, :self.grad_step],
                    actions[:, :self.grad_step],
                    prev_obs_acts[:, :self.grad_step],
                    delta_obs_rews[:, :self.grad_step],
                    masks[:, :self.grad_step]
                )
                for k, v in train_loss.items():
                    train_losses[k] = [v]
                
                update_times = int(1000 / self.grad_step)

                for i in range(1, update_times):
                    if masks[:, self.grad_step * i:self.grad_step * (i + 1)].sum().cpu().numpy() < batch_size:
                        break
                    train_loss, h_states = self.train_onestep(
                        obss[:, self.grad_step * i:self.grad_step * (i + 1)],
                        actions[:, self.grad_step * i:self.grad_step * (i + 1)],
                        prev_obs_acts[:, self.grad_step * i:self.grad_step * (i + 1)],
                        delta_obs_rews[:, self.grad_step * i:self.grad_step * (i + 1)],
                        masks[:, self.grad_step * i:self.grad_step * (i + 1)],
                        h_states
                    )
                    for k, v in train_loss.items():
                        train_losses[k].append(v)
            
            # log loss
            for k, v in train_losses.items():
                logger.logkv(f"train_loss/{k}", np.mean(v))
            new_holdout_loss = self.validate(validate_loader)
            for k, v in new_holdout_loss.items():
                logger.logkv(f"holdout_loss/{k}", v)

            if (epoch - 1) % eval_freq == 0:
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

            improvement = (holdout_loss - new_holdout_loss["dynamics_loss"]) / holdout_loss
            if improvement > 0.01:
                holdout_loss = new_holdout_loss["dynamics_loss"]
                best_state_dict = deepcopy(self.state_dict())
                cnt = 0
            else:
                cnt += 1
            if max_epoch is not None:
                if epoch >= max_epoch:
                    break
            else:
                if cnt >= 5:
                    break

        self.load_state_dict(best_state_dict)
        torch.save(self.state_dict(), os.path.join(logger.model_dir, "dynamics.pth"))

    def train_onestep(self, obss, actions, prev_obs_acts, delta_obs_rews, masks, h_states=None):
        obss = obss.reshape(-1, obss.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        obs_acts = torch.cat([obss, actions], -1)
        delta_obs_rews = delta_obs_rews.reshape(-1, delta_obs_rews.shape[-1])

        embeddings, h_states = self.policy_encoder(prev_obs_acts, h_states)
        if self.stop_dynamics_grad_to_encoder:
            dynamics_mean, dynamics_logvar = self.dynamics_model(obs_acts, embeddings.detach())
        else:
            dynamics_mean, dynamics_logvar = self.dynamics_model(obs_acts, embeddings)
        policy_dist = self.policy_decoder(obss, embeddings)

        # dynamics loss
        if self.deterministic:
            dynamics_loss = ((dynamics_mean - delta_obs_rews) ** 2).mean(-1).flatten()[masks.flatten() > 0].mean()
        else:
            inv_var = torch.exp(-dynamics_logvar)
            mse_loss_inv = (torch.pow(dynamics_mean - delta_obs_rews, 2) * inv_var).mean(-1)
            var_loss = dynamics_logvar.mean(-1)
            dynamics_loss = mse_loss_inv + var_loss
            dynamics_loss = dynamics_loss.flatten()[masks.flatten() > 0].mean()
            dynamics_loss += (0.01 * self.dynamics_model.max_logvar.sum() - 0.01 * self.dynamics_model.min_logvar.sum())

        # policy recon loss
        # policy_recon_loss = - torch.clamp(
        #     policy_dist.log_prob(actions).flatten()[masks.flatten() > 0], None, np.log(self.clip_val)
        # ).mean()
        policy_recon_loss = -policy_dist.log_prob(actions).flatten()[masks.flatten() > 0].mean()
        
        # pred_actions, _ = policy_dist.rsample()
        # policy_recon_loss = ((actions - pred_actions) ** 2).mean(-1).flatten()[masks.flatten()>0].mean()

        total_loss = dynamics_loss + \
            self.policy_recon_weight * policy_recon_loss + self.policy_encoder.get_decay_loss()

        self.dynamics_model_optim.zero_grad()
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()

        total_loss.backward()

        self.dynamics_model_optim.step()
        self.encoder_optim.step()
        self.decoder_optim.step()

        return {
            "dynamics_loss": dynamics_loss.item(),
            "policy_recon_loss": policy_recon_loss.item()
        }, h_states.detach()

    @torch.no_grad()
    def validate(self, validate_loader):
        self.eval()
        dynamics_losses, policy_recon_losses = [], []
        for obss, actions, prev_obs_acts, \
            delta_obs_rews, rtgs, stgs, masks in validate_loader:

            obss = obss.reshape(-1, obss.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
            obs_acts = torch.cat([obss, actions], -1)
            delta_obs_rews = delta_obs_rews.reshape(-1, delta_obs_rews.shape[-1])

            embeddings, _ = self.policy_encoder(prev_obs_acts)
            dynamics_mean, dynamics_logvar = self.dynamics_model(obs_acts, embeddings)
            policy_dist = self.policy_decoder(obss, embeddings)
            pred_actions, _ = policy_dist.mode()

            dynamics_loss = ((dynamics_mean - delta_obs_rews) ** 2).mean(-1).flatten()[masks.flatten() > 0].mean()
            policy_recon_loss = ((pred_actions - actions) ** 2).mean(-1).flatten()[masks.flatten() > 0].mean()

            dynamics_losses.append(dynamics_loss.item())
            policy_recon_losses.append(policy_recon_loss.item())

        return {
            "dynamics_loss": np.mean(dynamics_losses),
            "policy_recon_loss": np.mean(policy_recon_losses)
        }

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
        self.eval()
        fake_ep_info = []
        obs_dim = self.policy_decoder.input_dim
        action_dim = self.policy_decoder.dist.output_dim
        obs = self.real_env.reset().reshape(1, obs_dim)

        num_episodes = 0
        value, episode_length = 0, 0
        last_obs_act = np.zeros((1, obs_dim+action_dim), dtype=np.float32)
        h_state = None

        while num_episodes < num_eval_episodes:
            action = policy.select_action(obs, deterministic=False)
            next_obs, reward, terminal, info = self.step(obs, action, last_obs_act, h_state)
            reward, terminal = reward.flatten()[0], terminal.flatten()[0]
            value += reward * (gamma ** episode_length)
            episode_length += 1
            last_obs_act = np.concatenate([obs, action], axis=-1).reshape(1, -1)
            h_state = info["h_state"]
            obs = next_obs.copy()
            if terminal or episode_length >= 1000:
                fake_ep_info.append(
                    {"value": value, "episode_length": episode_length}
                )
                num_episodes += 1
                value, episode_length = 0, 0
                obs = self.real_env.reset().reshape(1, obs_dim)
                last_obs_act = np.zeros((1, obs_dim+action_dim), dtype=np.float32)
                h_state = None

        fake_value = np.mean([info["value"] for info in fake_ep_info])

        return fake_value

    def state_dict(self):
        return {
            "dynamics_model": self.dynamics_model.state_dict(),
            "policy_encoder": self.policy_encoder.state_dict(),
            "policy_decoder": self.policy_decoder.state_dict(),
            "scaler/mu": torch.as_tensor(np.array(self.scaler.mu)),
            "scaler/std": torch.as_tensor(np.array(self.scaler.std))
        }

    def load_state_dict(self, state_dict):
        self.dynamics_model.load_state_dict(state_dict["dynamics_model"])
        self.policy_encoder.load_state_dict(state_dict["policy_encoder"])
        self.policy_decoder.load_state_dict(state_dict["policy_decoder"])
        self.scaler.mu = state_dict["scaler/mu"].cpu().numpy().flatten(),
        self.scaler.std = state_dict["scaler/std"].cpu().numpy().flatten()