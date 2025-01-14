"""
WARNING: This script is only for instructive purposes, to point out different portions
         of the config -- the preferred way to launch training runs is still with external
         jsons and scripts/train.py (and optionally using scripts/hyperparameter_helper.py
         to generate several config jsons by sweeping config settings). See the online
         documentation for more information about launching training.

Example script for training a BC-RNN agent by manually setting portions of the config in 
python code.
"""
import argparse
import os

import robomimic
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.macros as Macros
from robomimic.config import config_factory
from robomimic.scripts.train import train


def r2d2_hyperparameters(config, args):
    """
    Sets R2D2-specific hyperparameters.

    Args:
        config (Config): Config to modify
        args: Command-line arguments passed in to this script.

    Returns:
        Config: Modified config
    """
    ## save config - if and when to save checkpoints ##
    config.experiment.save.enabled = True                       # whether model saving should be enabled or disabled
    config.experiment.save.every_n_seconds = None               # save model every n seconds (set to None to disable)
    config.experiment.save.every_n_epochs = int(args.num_epochs / 20) # save model every n epochs (set to None to disable)
    config.experiment.save.epochs = []                          # save model on these specific epochs
    config.experiment.save.on_best_validation = False           # save models that achieve best validation score
    config.experiment.save.on_best_rollout_return = False       # save models that achieve best rollout return
    config.experiment.save.on_best_rollout_success_rate = True  # save models that achieve best success rate

    # epoch definition - if not None, set an epoch to be this many gradient steps, else the full dataset size will be used
    config.experiment.epoch_every_n_steps = None                  # None -> epoch is full dataset
    config.experiment.validation_epoch_every_n_steps = None       # None -> epoch is full dataset

    # envs to evaluate model on (assuming rollouts are enabled), to override the metadata stored in dataset
    config.experiment.env = None                                # no need to set this (unless you want to override)
    config.experiment.additional_envs = None                    # additional environments that should get evaluated

    ## rendering config ##
    config.experiment.render = False                            # render on-screen or not
    config.experiment.render_video = True                       # render evaluation rollouts to videos
    config.experiment.keep_all_videos = False                   # save all videos, instead of only saving those for saved model checkpoints
    config.experiment.video_skip = 5                            # render video frame every n environment steps during rollout

    ## evaluation rollout config ##
    config.experiment.rollout.enabled = False                    # enable evaluation rollouts
    config.experiment.rollout.n = 50                            # number of rollouts per evaluation
    config.experiment.rollout.horizon = 400                     # set horizon based on length of demonstrations (can be obtained with scripts/get_dataset_info.py)
    config.experiment.rollout.rate = 50                         # do rollouts every @rate epochs
    config.experiment.rollout.warmstart = 0                     # number of epochs to wait before starting rollouts
    config.experiment.rollout.terminate_on_success = True       # end rollout early after task success

    ## dataset loader config ##

    # num workers for loading data - generally set to 0 for low-dim datasets, and 2 for image datasets
    config.train.num_data_workers = 0 if args.use_ram else len(os.sched_getaffinity(0)) # num CPU cores available to current training job -- do NOT use os.cpu_count()! -- source: https://stackoverflow.com/a/55423170

    # One of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5 in memory - this is
    # by far the fastest for data loading. Set to "low_dim" to cache all non-image data. Set
    # to None to use no caching - in this case, every batch sample is retrieved via file i/o.
    # You should almost never set this to None, even for large image datasets.
    config.train.hdf5_cache_mode = "all"

    config.train.hdf5_use_swmr = True                           # used for parallel data loading

    # if true, normalize observations at train and test time, using the global mean and standard deviation
    # of each observation in each dimension, computed across the training set. See SequenceDataset.normalize_obs
    # in utils/dataset.py for more information.
    config.train.hdf5_normalize_obs = True

    # if provided, demonstrations are filtered by the list of demo keys under "mask/@hdf5_filter_key"
    config.train.hdf5_filter_key = "train"                      # by default, use "train" and "valid" filter keys corresponding to train-valid split
    config.train.hdf5_validation_filter_key = "valid"

    # fetch sequences of length 15 from dataset for RNN training
    config.train.seq_length = 15

    # keys from hdf5 to load per demonstration, besides "obs" and "next_obs"
    config.train.dataset_keys = (
        "actions",
        "rewards",
        "dones",
    )

    # one of [None, "last"] - set to "last" to include goal observations in each batch
    config.train.goal_mode = None                               # no need for goal observations

    ## learning config ##
    config.train.cuda = True                                    # try to use GPU (if present) or not
    config.train.batch_size = args.batch_size                               # batch size
    config.train.num_epochs = args.num_epochs                              # number of training epochs
    config.train.seed = args.seed                                       # seed for training


    ### Observation Config ###
    config.observation.modalities.obs.low_dim = []                          # no low-dim obs
    config.observation.modalities.obs.rgb = ['robot0_eye_in_hand_image']    # wrist camera image observations
    config.observation.modalities.goal.low_dim = []                         # no low-dim goals
    config.observation.modalities.goal.rgb = []                             # no image goals

    # observation encoder architecture - applies to all networks that take observation dicts as input

    config.observation.encoder.rgb.core_class = "VisualCore"
    config.observation.encoder.rgb.core_kwargs.feature_dimension = args.encoder_features_dim # image encoder output dimension
    config.observation.encoder.rgb.core_kwargs.backbone_class = 'ResNet18Conv'                         # ResNet backbone for image observations (unused if no image observations)
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = False                # kwargs for visual core
    config.observation.encoder.rgb.core_kwargs.backbone_kwargs.input_coord_conv = False
    config.observation.encoder.rgb.core_kwargs.pool_class = "SpatialSoftmax"                # Alternate options are "SpatialMeanPool" or None (no pooling)
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.num_kp = 64                      # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.learnable_temperature = False    # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.temperature = 1.0                # Default arguments for "SpatialSoftmax"
    config.observation.encoder.rgb.core_kwargs.pool_kwargs.noise_std = 0.0                  # Default arguments for "SpatialSoftmax"

    # if you prefer to use pre-trained visual representations, uncomment the following lines
    # R3M
    # config.observation.encoder.rgb.core_kwargs.backbone_class = 'R3MConv'                         # R3M backbone for image observations (unused if no image observations)
    # config.observation.encoder.rgb.core_kwargs.backbone_kwargs.r3m_model_class = 'resnet18'       # R3M model class (resnet18, resnet34, resnet50)
    # config.observation.encoder.rgb.core_kwargs.backbone_kwargs.freeze = True                      # whether to freeze network during training or allow finetuning
    # config.observation.encoder.rgb.core_kwargs.pool_class = None                                  # no pooling class for pretraining model
    # MVP
    # config.observation.encoder.rgb.core_kwargs.backbone_class = 'MVPConv'                                   # MVP backbone for image observations (unused if no image observations)
    # config.observation.encoder.rgb.core_kwargs.backbone_kwargs.mvp_model_class = 'vitb-mae-egosoup'         # MVP model class (vits-mae-hoi, vits-mae-in, vits-sup-in, vitb-mae-egosoup, vitl-256-mae-egosoup)
    # config.observation.encoder.rgb.core_kwargs.backbone_kwargs.freeze = True                                # whether to freeze network during training or allow finetuning
    # config.observation.encoder.rgb.core_kwargs.pool_class = None                                            # no pooling class for pretraining model

    # observation randomizer class - set to None to use no randomization, or 'CropRandomizer' to use crop randomization
    config.observation.encoder.rgb.obs_randomizer_class = None

    # kwargs for observation randomizers (for the CropRandomizer, this is size and number of crops)
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_height = 256
    config.observation.encoder.rgb.obs_randomizer_kwargs.crop_width = 256
    config.observation.encoder.rgb.obs_randomizer_kwargs.num_crops = 1
    config.observation.encoder.rgb.obs_randomizer_kwargs.pos_enc = False

    ### Algo Config ###

    # optimization parameters
    config.algo.optim_params.policy.learning_rate.initial = args.lr        # policy learning rate
    config.algo.optim_params.policy.learning_rate.decay_factor = 0.1    # factor to decay LR by (if epoch schedule non-empty)
    config.algo.optim_params.policy.learning_rate.epoch_schedule = []   # epochs where LR decay occurs
    config.algo.optim_params.policy.regularization.L2 = 0.00            # L2 regularization strength

    # loss weights
    config.algo.loss.l2_weight = 1.0    # L2 loss weight
    config.algo.loss.l1_weight = 0.0    # L1 loss weight
    config.algo.loss.cos_weight = 0.0   # cosine loss weight

    # MLP network architecture (layers after observation encoder and RNN, if present)
    config.algo.actor_layer_dims = ()   # empty MLP - go from RNN layer directly to action output

    # stochastic GMM policy
    config.algo.gmm.enabled = True                      # enable GMM policy - policy outputs GMM action distribution
    config.algo.gmm.num_modes = 5                       # number of GMM modes
    config.algo.gmm.min_std = 0.0001                    # minimum std output from network
    config.algo.gmm.std_activation = "softplus"         # activation to use for std output from policy net
    config.algo.gmm.low_noise_eval = True               # low-std at test-time

    # rnn policy config
    config.algo.rnn.enabled = True      # enable RNN policy
    config.algo.rnn.horizon = config.train.seq_length        # unroll length for RNN - should usually match train.seq_length
    config.algo.rnn.hidden_dim = args.rnn_hidden_dim    # hidden dimension size
    config.algo.rnn.rnn_type = "LSTM"   # rnn type - one of "LSTM" or "GRU"
    config.algo.rnn.num_layers = 2      # number of RNN layers that are stacked
    config.algo.rnn.open_loop = args.open_loop   # if True, action predictions are only based on a single observation (not sequence) + hidden state
    config.algo.rnn.kwargs.bidirectional = False          # rnn kwargs

    return config


def get_config(args):
    """Construct config for training."""
    # handle args
    if args.data_dir is None:
        raise ValueError("Please provide an argument for --data_dir")

    if args.output_dir is None:
        raise ValueError("Please provide an argument for --output_dir")

    # make default BC config
    config = config_factory(algo_name="bc")

    ### Experiment Config ###
    config.experiment.validate = True                           # whether to do validation or not
    config.experiment.logging.terminal_output_to_txt = False    # whether to log stdout to txt file 
    config.experiment.logging.log_tb = True                     # enable tensorboard logging

    ### Train Config ###
    config.train.data = args.data_dir                            # path(s) to expert demos dataset(s)

    # Write all results to this directory. A new folder with the timestamp will be created
    # in this directory, and it will contain three subfolders - "log", "models", and "videos".
    # The "log" directory will contain tensorboard and stdout txt logs. The "models" directory
    # will contain saved model checkpoints. The "videos" directory contains evaluation rollout
    # videos.
    config.train.output_dir = args.output_dir                        # path to output folder

    # Load default hyperparameters based on dataset type
    config = r2d2_hyperparameters(config, args)

    # Shrink training and rollout times to test a full training run quickly.
    if args.debug:

        # train and validate for 3 gradient steps per epoch, and 2 total epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # rollout and model saving every epoch, and make rollouts short
        config.experiment.save.every_n_epochs = 1
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

    return config


def str_to_bool(s: str) -> bool:
    if s not in {'True', 'False'}:
        raise ValueError('Invalid boolean string argument given.')
    return s == 'True'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # debug flag for quick training run
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    # Args for UPGM R2D2
    parser.add_argument("--data_dir", type=str, default='R2D2/data',
                        help="Directory containing the expert demonstrations used for training.")
    parser.add_argument("--cam_serial_num", type=str, default='138422074005',
                        help="Serial number of the camera used to record videos of the demonstration trajectories.")
    parser.add_argument("--output_dir", type=str, default='logs/robomimic/bc-rnn/sample-run',
                        help="Logs directory for TensorBoard stats and policy demo gifs.")
    parser.add_argument("--checkpoint_dir", type=str,
                        help="Directory containing the saved checkpoint.")
    parser.add_argument("--checkpoint_epoch", type=str, default='',
                        help="The epoch number at which to resume training. If 0 (represented by ''), start fresh.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=10000,
                        help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per gradient step.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for training.")
    parser.add_argument("--load_optimizer", type=str_to_bool, default=False,
                        help="(Only applicable when loading checkpoint) Whether to load the previously saved optimizer state.")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Size of (square) image observations.")
    # parser.add_argument("--image_encoder", type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'],
    #                     help="Which image encoder to use for the BC policy.")
    parser.add_argument("--apply_aug", type=str_to_bool, default=False,
                        help="Whether to use standard data augmentations on the training set (e.g., random crop).")
    parser.add_argument("--spartn", type=str_to_bool, default=False,
                        help="Whether to use SPARTN data augmentations on the training set.")
    parser.add_argument("--use_ram", type=str_to_bool, default=False,
                        help="Whether to load all training data into memory instead of reading from disk (for small datasets).")
    parser.add_argument("--checkpoint_epoch_offset", type=str_to_bool, default=True,
                        help="(Only applicable when loading checkpoint) If True, the starting epoch number is 0. Else, we start where the previous checkpoint finished.")
    parser.add_argument("--encoder_features_dim", type=int, default=512,
                        help="Output size of image encoder.")
    parser.add_argument("--rnn_hidden_dim", type=int, default=400,
                        help="Hidden size of recurrent neural network policy backbone.")
    parser.add_argument("--open_loop", type=str_to_bool, default=False,
                        help="If True, action predictions are only based on a single observation (not sequence) + hidden state.")

    args = parser.parse_args()

    # Turn debug mode on possibly
    if args.debug:
        Macros.DEBUG = True

    # config for training
    config = get_config(args)

    # set torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # run training
    train(config, args, device=device)
