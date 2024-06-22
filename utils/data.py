import torch
import numpy as np
import collections
import scipy.signal


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
        x1,
        x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
        x1 + discount * x2,
        x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device="cpu"):
        super().__init__()

        rtgs = []
        rewards_ = []
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        for i in range(dataset["rewards"].shape[0]):
            rewards_.append(dataset["rewards"][i])
            episode_step += 1
            terminal = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000)
            if terminal or final_timestep:
                episode_step = 0
                rewards_ = np.array(rewards_)
                rtgs_ = discount_cumsum(rewards_, 1)
                rtgs.append(rtgs_)
                rewards_ = []

        self.observations = np.array(dataset["observations"], dtype=np.float32)
        self.next_observations = np.array(dataset["next_observations"], dtype=np.float32)
        self.actions = np.array(dataset["actions"], dtype=np.float32)
        self.rewards = np.array(dataset["rewards"], dtype=np.float32)
        self.rtgs = np.concatenate(rtgs, dtype=np.float32).reshape(-1, 1)

        self.obs_mean = self.observations.mean(0)
        self.obs_std = self.observations.std(0) + 1e-6
        self.rtg_min, self.rtg_max = self.rtgs.min(), self.rtgs.max()
        print(f"rtg min: {self.rtg_min}, rtg max: {self.rtg_max}")

        self.device = torch.device(device)
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs, act, next_obs, reward, rtg = self.observations[idx], self.actions[idx], \
            self.next_observations[idx], self.rewards[idx], self.rtgs[idx]
        delta_obs = (next_obs - obs).copy()
        obs = (obs - self.obs_mean) / self.obs_std
        rtg = (rtg - self.rtg_min) / (self.rtg_max - self.rtg_min)
        obs = torch.from_numpy(obs).to(self.device)
        act = torch.from_numpy(act).to(self.device)
        rtg = torch.from_numpy(rtg).to(self.device)
        delta_obs_rew = torch.from_numpy(np.concatenate([delta_obs, reward])).to(self.device)
        return obs, act, rtg, delta_obs_rew


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, traj_len=1000, device="cpu"):
        super().__init__()

        self.obs_dim = dataset["observations"].shape[-1]
        self.action_dim = dataset["actions"].shape[-1]
        self.traj_len = traj_len
        self.device = torch.device(device)

        self.obs_mean = dataset["observations"].mean(0)
        self.obs_std = dataset["observations"].std(0) + 1e-6

        data_ = collections.defaultdict(list)
        
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        self.trajs = []
        for i in range(dataset["rewards"].shape[0]):
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                data_[k].append(dataset[k][i])
            episode_step += 1
            terminal = bool(dataset['terminals'][i])
            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == 1000)
            if terminal or final_timestep:
                episode_step = 0
                episode_data = {}
                rtg = np.array(data_['rewards'])
                rtg = discount_cumsum(rtg, 1)
                stg = np.array(data_['observations'])
                steps = np.arange(stg.shape[0], 0, -1)
                stg = stg / np.expand_dims(steps, axis=-1)
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                episode_data['rtg'] = rtg
                episode_data['stg'] = stg
                self.trajs.append(episode_data)
                data_ = collections.defaultdict(list)

        self.indices = np.arange(len(self.trajs))
    
        returns = np.array([np.sum(t['rewards']) for t in self.trajs])
        num_samples = np.sum([t['rewards'].shape[0] for t in self.trajs])
        print(f'Number of samples collected: {num_samples}')
        print(f'Num trajectories: {len(self.trajs)}')
        print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

        rtgs = np.concatenate([t['rtg'] for t in self.trajs], axis=0)
        stgs = np.concatenate([t['stg'] for t in self.trajs], axis=0)

        self.rtg_min = rtgs.min(0)
        self.rtg_max = rtgs.max(0)
        self.stg_min = stgs.min(0)
        self.stg_max = stgs.max(0)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_idx = self.indices[idx]
        traj = self.trajs[traj_idx].copy()
        raw_obss = traj['observations']
        actions = traj['actions']
        next_obss = traj['next_observations']
        rewards = traj['rewards'].reshape(-1, 1)
        rtgs = traj['rtg'].reshape(-1, 1)
        stgs = traj['stg']
        
        rtgs = (rtgs - self.rtg_min) / (self.rtg_max - self.rtg_min)
        stgs = (stgs - self.stg_min) / (self.stg_max - self.stg_min)
        delta_obss = next_obss - raw_obss
        obss = (raw_obss - self.obs_mean) / self.obs_std
    
        # padding
        tlen = obss.shape[0]
        obss = np.concatenate([obss, np.zeros((self.traj_len - tlen, self.obs_dim))], axis=0)
        actions = np.concatenate([actions, np.zeros((self.traj_len - tlen, self.action_dim))], axis=0)
        obs_acts = np.concatenate([obss, actions], axis=-1)
        prev_obs_acts = np.concatenate([np.zeros_like(obs_acts[0:1]), obs_acts[:-1]], axis=0)
        delta_obs_rews = np.concatenate([delta_obss, rewards], axis=-1)
        delta_obs_rews = np.concatenate([delta_obs_rews, np.zeros((self.traj_len - tlen, self.obs_dim+1))], axis=0)
        masks = np.concatenate([np.ones(tlen), np.zeros(self.traj_len - tlen)], axis=0)
        rtgs = np.concatenate([rtgs, np.zeros((self.traj_len - tlen, rtgs.shape[-1]))], axis=0)
        stgs = np.concatenate([stgs, np.zeros((self.traj_len - tlen, stgs.shape[-1]))], axis=0)

        obss = torch.from_numpy(obss).to(dtype=torch.float32, device=self.device)
        actions = torch.from_numpy(actions).to(dtype=torch.float32, device=self.device)
        prev_obs_acts = torch.from_numpy(prev_obs_acts).to(dtype=torch.float32, device=self.device)
        delta_obs_rews = torch.from_numpy(delta_obs_rews).to(dtype=torch.float32, device=self.device)
        rtgs = torch.from_numpy(rtgs).to(dtype=torch.float32, device=self.device)
        stgs = torch.from_numpy(stgs).to(dtype=torch.float32, device=self.device)
        masks = torch.from_numpy(masks).to(dtype=torch.long, device=self.device)

        return obss, actions, prev_obs_acts, delta_obs_rews, rtgs, stgs, masks


if __name__ == '__main__':
    import d4rl
    import gym

    dataset = gym.make("walker2d-medium-replay-v2").get_dataset()
    dataset = SequenceDataset(dataset)

    obss, actions, prev_obs_acts, delta_obs_rews, rtgs, stgs, masks = \
        dataset.__getitem__(0)
    print(obss)
    print(prev_obs_acts[:, :17])
    # print(rtgs.shape)
    # print(stgs.shape)
    # print(masks.shape)