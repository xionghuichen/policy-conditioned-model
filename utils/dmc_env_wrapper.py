import numpy as np
import tensorflow as tf


class DMCEnvWrapper():
    def __init__(self, env, keys) -> None:
        self.env = env
        self.keys = keys
    
    def reset(self):
        timestep = self.env.reset()
        observation = []
        for key in self.keys:
            data = timestep.observation[key]
            if len(data.shape) > 1:
                data = data.flatten()
            if len(data.shape) == 0:
                data = np.array([data])
            observation.append(data)
        observation = np.concatenate(observation, 0)
        return observation
    
    def step(self, action):
        timestep = self.env.step(action)
        observation = []
        for key in self.keys:
            data = timestep.observation[key]
            if len(data.shape) > 1:
                data = data.flatten()
            if len(data.shape) == 0:
                data = np.array([data])
            observation.append(data)
        observation = np.concatenate(observation, 0)
        terminal = (timestep.step_type > 1)
        return observation, timestep.reward, terminal, {}


def cartpole_swingup_obs_map(obs):
    if len(obs.shape) == 1:
        obs = obs[None, :]
    obs = {
        'position': obs[:, :3],
        'velocity': obs[:, 3:],
    }
    return obs

def cheetah_run_obs_map(obs):
    if len(obs.shape) == 1:
        obs = obs[None, :]
    obs = {
        'position': obs[:, :8],
        'velocity': obs[:, 8:],
    }
    return obs

def finger_turn_hard_obs_map(obs):
    if len(obs.shape) == 1:
        obs = obs[None, :]
    obs = {
        'dist_to_target': obs[:, 0:1], 
        'position': obs[:, 1:5],
        'target_position': obs[:, 5:7],
        'touch': obs[:, 7:9],
        'velocity': obs[:, 9:]
    }
    return obs

def fish_swim_obs_map(obs):
    if len(obs.shape) == 1:
        obs = obs[None, :]
    obs = {
        'joint_angles': obs[:, :7],
        'target': obs[:, 7:10],
        'upright': obs[:, 10:11],
        'velocity': obs[:, 11:],
    }
    return obs

def humanoid_run_obs_map(obs):
    if len(obs.shape) == 1:
        obs = obs[None, :]
    obs = {
        'com_velocity': obs[:, 0:3],
        'extremities': obs[:, 3:15],
        'head_height': obs[:, 15:16],
        'joint_angles': obs[:, 16:37],
        'torso_vertical': obs[:, 37:40],
        'velocity': obs[:, 40:]
    }
    return obs

def manipulator_insert_ball_obs_map(obs):
    if len(obs.shape) == 1:
        obs = obs[None, :]
    obs = {
        'arm_pos': obs[:, 0:16].reshape(-1, 8, 2),
        'arm_vel': obs[:, 16:24],
        'hand_pos': obs[:, 24:28],
        'object_pos': obs[:, 28:32],
        'object_vel': obs[:, 32:35],
        'target_pos': obs[:, 35:39],
        'touch': obs[:, 39:]
    }
    return obs

def walker_stand_obs_map(obs):
    if len(obs.shape) == 1:
        obs = obs[None, :]
    obs = {
        "height": obs[:, 0:1],
        "orientations": obs[:, 1:15],
        "velocity": obs[:, 15:]
    }
    return obs

def walker_walk_obs_map(obs):
    if len(obs.shape) == 1:
        obs = obs[None, :]
    obs = {
        "height": obs[:, 0:1],
        "orientations": obs[:, 1:15],
        "velocity": obs[:, 15:]
    }
    return obs


def get_obs_map_fn(task):
    if 'cartpole_swingup' in task:
        return cartpole_swingup_obs_map
    if 'cheetah_run' in task:
        return cheetah_run_obs_map
    if 'finger_turn_hard' in task:
        return finger_turn_hard_obs_map
    if 'fish_swim' in task:
        return fish_swim_obs_map
    if 'humanoid_run' in task:
        return humanoid_run_obs_map
    if 'manipulator_insert_ball' in task:
        return manipulator_insert_ball_obs_map
    if 'walker_stand' in task:
        return walker_stand_obs_map
    if 'walker_walk' in task:
        return walker_walk_obs_map


if __name__ == '__main__':
    from utils.dm_control_suite import ControlSuite
    task = ControlSuite("cheetah_run")
    env = DMCEnvWrapper(task.environment)
    print()