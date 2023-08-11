"""
The main script for evaluating a BC-RNN policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""
import argparse
import json
import h5py
import imageio
import numpy as np
import os
import time
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

from r2d2.robot_env import RobotEnv
from upgm.utils.data_utils import process_img


def unscale_action(action):
    # Undoes the action scaling at training time by reversing the formula here: https://stats.stackexchange.com/a/178629
    # Note: The gripper action doesn't need to be scaled at test time, as anything below 0 is 'close' and anything above 0 is 'open'.
    min_delta_pos, max_delta_pos = -0.20, 0.20
    min_delta_euler, max_delta_euler = -0.20, 0.20
    action[:3] = (action[:3] + 1) * (max_delta_pos - min_delta_pos) / 2 + min_delta_pos
    action[3:6] = (action[3:6] + 1) * (max_delta_euler - min_delta_euler) / 2 + min_delta_euler
    return action


def rollout(policy, env, horizon, target_label, render=False, video_writer=None, video_skip=5, return_obs=False):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.

    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        target_label (str): Language annotation of the target ibhect,
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu. 
            They are excluded by default because the low-dimensional simulation states should be a minimal 
            representation of the environment.

    Returns:
        traj (dict): dictionary that corresponds to the rollout trajectory
    """
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    env.reset()
    input('Press Enter to begin...')
    state_dict = env.get_state()

    results = {}
    video_count = 0  # video frame counter
    traj = dict(actions=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[]))
    try:
        for step_i in range(horizon):
            try:
                # Mark starting time.
                step_start_time = time.time()

                # Get environment observations.
                state_dict = env.get_state()
                env_obs_dict = env.get_observation()
                image = process_img(env_obs_dict['image'][args.cam_serial_num][:], args.img_size) # shape: (H, W, 3)
                image = np.transpose(image, (2, 0, 1)) # shape: (3, H, W)
                image = image / 255.0 - 0.5 # normalize images to [-0.5, 0.5]
                obs = {'robot0_eye_in_hand_image': image, 'target_label': target_label,}
                if 'target_label' in obs:
                    target_label_inputs = obs.pop('target_label') # IMPORTANT TODO: DO SOMETHING WITH THE LANGUAGE INPUTS!!!

                # Get the action from the policy.
                action = policy(ob=obs)
                action = unscale_action(action)
                action = np.clip(action, -1, 1)
                np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
                print('action:', action)

                # Perform the action.
                action_info = env.step(action)

                if video_writer is not None:
                    if video_count % video_skip == 0:
                        video_img = []
                        for cam_name in camera_names:
                            video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                        video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                        video_writer.append_data(video_img)
                    video_count += 1

                # collect transition
                traj["actions"].append(action)
                traj["states"].append(state_dict)
                if return_obs:
                    # Note: We need to "unprocess" the observations to prepare to write them to dataset.
                    #       This includes operations like channel swapping and float to uint8 conversion
                    #       for saving disk space.
                    traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))

                # Sleep the amount necessary to maintain consistent control frequency.
                elapsed_time = time.time() - step_start_time
                time_to_sleep = (1 / env.control_hz) - elapsed_time
                print('time_to_sleep:', time_to_sleep)
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                else:
                    print(f'WARNING: Environment step took longer than expected.')

            except KeyboardInterrupt:
                user_input = input('\nEnter (q) (Ctrl-C) to quit the program, or anything else to continue to the next episode...')
                if user_input == 'q':
                    quit_early = True
                    break
                else:
                    break

    except Exception as e:
        print("WARNING: got exception {}".format(e))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def run_trained_agent(args):
    # path to model checkpoint
    ckpt_path = os.path.join(args.checkpoint_dir, 'models', f'model_epoch_{args.checkpoint_epoch}.pth')

    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

    # read rollout settings
    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        # read horizon from config
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment from saved checkpoint
    env = RobotEnv(action_space='cartesian_velocity')

    # maybe set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # # maybe create video writer
    video_writer = None
    write_video = args.video_path is not None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    target_label = ''
    for i in range(rollout_num_episodes):
        # Prompt user for target label.
        if target_label == '':
            target_label = input("Enter the target object to grasp: ")
        else:
            user_input = input("Enter the target object to grasp. To repeat the previous target object, press Enter without typing anything: ")
            if user_input != '':
                target_label = user_input
        print(f'Target object to grasp: {target_label}')
        traj = rollout(
            policy=policy, 
            env=env, 
            horizon=rollout_horizon,
            target_label=target_label,
            render=False,
            video_writer=video_writer, 
            video_skip=args.video_skip, 
            return_obs=(write_dataset and args.dataset_obs),
        )

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(i))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    # if write_video:
    #     video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # number of rollouts
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=27,
        help="number of rollouts",
    )

    # maximum horizon of rollout, to override the one stored in the model checkpoint
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override maximum horizon of rollout from the one in the checkpoint",
    )

    # Dump a video of the rollouts to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render rollouts to this video file path",
    )

    # How often to write video frames during the rollout
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
    )

    # for seeding before starting rollouts
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="(optional) set seed for rollouts",
    )

    # Args for UPGM R2D2
    parser.add_argument("--checkpoint_dir", type=str,
                        help="Directory containing the saved checkpoint.")
    parser.add_argument("--checkpoint_epoch", type=str, default='',
                        help="The epoch number at which to resume training. If 0 (represented by ''), start fresh.")
    parser.add_argument("--cam_serial_num", type=str, default='138422074005',
                        help="Serial number of the camera used to record videos of the demonstration trajectories.")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Size of (square) image observations.")

    args = parser.parse_args()
    if args.checkpoint_dir is None or args.checkpoint_epoch == '':
        raise ValueError('Please provide valid --checkpoint_dir and --checkpoint_epoch arguments.')

    run_trained_agent(args)

