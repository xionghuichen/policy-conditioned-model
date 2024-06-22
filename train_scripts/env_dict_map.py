data_map ={
    "hopper-medium-replay-v2": {
        "load_dataset_path": None,
        "load_eval_policy_path": "hopper/hopper_online",
        "dataset_type": "d4rl"
    },
    "halfcheetah-medium-replay-v2": {
        "load_dataset_path": None,
        "load_eval_policy_path": "halfcheetah/halfcheetah_online",
        "dataset_type": "d4rl"
    },
    "walker2d-medium-replay-v2": {
        "load_dataset_path": None,
        "load_eval_policy_path": "walker/walker_online",
        "dataset_type": "d4rl"
    },
    "hopper-medium-expert-v2": {
        "load_dataset_path": None,
        "load_eval_policy_path": "hopper/hopper_online",
        "dataset_type": "d4rl"
    },
    "halfcheetah-medium-expert-v2": {
        "load_dataset_path": None,
        "load_eval_policy_path": "halfcheetah/halfcheetah_online",
        "dataset_type": "d4rl"
    },
    "walker2d-medium-expert-v2": {
        "load_dataset_path": None,
        "load_eval_policy_path": "walker/walker_online",
        "dataset_type": "d4rl"
    },

    "HalfCheetah-v3": {
        "load_dataset_path": "HalfCheetah-v3(10-100).hdf5",
        "load_eval_policy_path": "halfcheetah/halfcheetah_online",
        "dataset_type": "d4rl"
    },

    ###

    "HalfCheetah-v3(0-20)": {
        "task": "HalfCheetah-v3",
        "load_dataset_path": "../sac/offline_dataset/halfcheetah/HalfCheetah-v3(0-20).hdf5",
        "load_eval_policy_path": "halfcheetah/halfcheetah_online",
        "dataset_type": "d4rl"
    },

    "HalfCheetah-v3(0-50)": {
        "task": "HalfCheetah-v3",
        "load_dataset_path": "../sac/offline_dataset/halfcheetah/HalfCheetah-v3(0-50).hdf5",
        "load_eval_policy_path": "halfcheetah/halfcheetah_online",
        "dataset_type": "d4rl"
    },

    "HalfCheetah-v3(0-80)": {
        "task": "HalfCheetah-v3",
        "load_dataset_path": "../sac/offline_dataset/halfcheetah/HalfCheetah-v3(0-80).hdf5",
        "load_eval_policy_path": "halfcheetah/halfcheetah_online",
        "dataset_type": "d4rl"
    },

    ###

    "cartpole_swingup": {
        "load_dataset_path": "rl_unplugged_dataset/dm_control_suite/cartpole_swingup.hdf5",
        "load_eval_policy_path": "eval_policy_state_dict/cartpole_swingup",
        "dataset_type": "rlunplugged",
    },

    "cheetah_run": {
        "load_dataset_path": "rl_unplugged_dataset/dm_control_suite/cheetah_run.hdf5",
        "load_eval_policy_path": "eval_policy_state_dict/cheetah_run",
        "dataset_type": "rlunplugged",
    },

    "finger_turn_hard": {
        "load_dataset_path": "rl_unplugged_dataset/dm_control_suite/finger_turn_hard.hdf5",
        "load_eval_policy_path": "eval_policy_state_dict/finger_turn_hard",
        "dataset_type": "rlunplugged",
    },
    
    "fish_swim": {
        "load_dataset_path": "rl_unplugged_dataset/dm_control_suite/fish_swim.hdf5",
        "load_eval_policy_path": "eval_policy_state_dict/fish_swim",
        "dataset_type": "rlunplugged",
    },

    "humanoid_run": {
        "load_dataset_path": "rl_unplugged_dataset/dm_control_suite/humanoid_run.hdf5",
        "load_eval_policy_path": "eval_policy_state_dict/humanoid_run",
        "dataset_type": "rlunplugged",
    },

    "manipulator_insert_ball": {
        "load_dataset_path": "rl_unplugged_dataset/dm_control_suite/manipulator_insert_ball.hdf5",
        "load_eval_policy_path": "eval_policy_state_dict/manipulator_insert_ball",
        "dataset_type": "rlunplugged",
    },

    "walker_stand": {
        "load_dataset_path": "rl_unplugged_dataset/dm_control_suite/walker_stand.hdf5",
        "load_eval_policy_path": "eval_policy_state_dict/walker_stand",
        "dataset_type": "rlunplugged",
    },

    "walker_walk": {
        "load_dataset_path": "rl_unplugged_dataset/dm_control_suite/walker_walk.hdf5",
        "load_eval_policy_path": "eval_policy_state_dict/walker_walk",
        "dataset_type": "rlunplugged",
    },
}