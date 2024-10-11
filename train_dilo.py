
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import datetime
from absl import app, flags
from ml_collections import config_flags
from dataclasses import dataclass
from dataset_utils import make_env_and_dataset
from dataset_utils import SNSD4RLDataset,SNSD4RLMixedDataset, split_into_trajectories
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
# from pathlib import Path
# import hydra
# from omegaconf import DictConfig
import numpy as np
import torch
from trainer import TrainerSNS as TrainerDILO
import dilo_utils
from models import  TwinQ, ValueFunction, TwinV

from dilo import DILO
import json
from  policy import GaussianPolicy, DeterministicPolicy
import time
from logging_utils.logx import EpochLogger
torch.backends.cudnn.benchmark = True 

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'hopper-medium-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('exp_name', 'dump', 'Epoch logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('expert_trajectories', 200, 'Number of expert trajectories')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_boolean('double', True, 'Use double q-learning')
flags.DEFINE_integer('max_steps', int(3e5), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_float('lamda', 0.8, 'f-maximization strength')
flags.DEFINE_float('beta', 0.5, 'Mixture distribution ratio') # Only implemented for 0.5
flags.DEFINE_float('ita', 0.5, 'Orthogonalization strength')
flags.DEFINE_float('tau', 3.0/50, 'Policy Temperature')
flags.DEFINE_string('maximizer', 'smoothed_chi', 'Which f divergence to use')
flags.DEFINE_string('grad', 'full', 'Semi-gradient or full-gradient?')

config_flags.DEFINE_config_file(
    'config',
    'configs/mujoco_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)




def evaluate(agent, env,
             num_episodes, device,  verbose: bool = False,normalization_stats=None) :
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            if 'obs_mean' in normalization_stats:
                action = agent.act(torch.FloatTensor((observation-normalization_stats['obs_mean'])/normalization_stats['obs_std']).to(device), deterministic=True)
            else:
                action = agent.act(torch.FloatTensor(observation).to(device), deterministic=True)
            observation, _, done, info = env.step(action.detach().cpu().numpy())

        for k in stats.keys():
            stats[k].append(info['episode'][k]) 
            if verbose:
                v = info['episode'][k]
                print(f'{k}:{v}')

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0



def main(_):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(FLAGS.save_dir, ts_str)
    exp_id = f"results/dilo/{FLAGS.env_name}/{FLAGS.expert_trajectories}_expert/observations/" + FLAGS.exp_name
    log_folder = exp_id + '/'+FLAGS.exp_name+'_s'+str(FLAGS.seed) 
    logger_kwargs={'output_dir':log_folder, 'exp_name':FLAGS.exp_name}
    
    e_logger = EpochLogger(**logger_kwargs)
    write_config = {}

    # Iterate through all flags and add them to the dictionary
    for flag_name in FLAGS.flags_by_module_dict()[sys.argv[0]]:
        key = flag_name.serialize().split('=')[0]
        if 'config' in key:
            continue
        value = flag_name.value

        write_config[key] = value
    write_config.update(dict(FLAGS.config))
    with open(log_folder+"/config.json", "w") as outfile: 
        json.dump(write_config, outfile,indent=4)
    hparam_str_dict = dict(seed=FLAGS.seed, env=FLAGS.env_name)
    hparam_str = ','.join([
        '%s=%s' % (k, str(hparam_str_dict[k]))
        for k in sorted(hparam_str_dict.keys())
    ])
    os.makedirs(save_dir, exist_ok=True)

    env, dataset, expert_dataset, offline_min, offline_max, normalization_stats = make_env_and_dataset(FLAGS.env_name, FLAGS.seed,FLAGS.expert_trajectories,normalize_obs=False)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    if FLAGS.grad=='semi':
        agent = DILO(qf=TwinQ(state_dim=obs_dim,act_dim=obs_dim),vf = TwinV(state_dim=obs_dim),policy=GaussianPolicy(obs_dim,act_dim),
                                                        optimizer_factory=torch.optim.Adam,
                                                        lamda=FLAGS.lamda, maximizer=FLAGS.maximizer,beta=FLAGS.beta,ita=FLAGS.ita,tau=FLAGS.tau, gradient_type=FLAGS.grad, lr=3e-4, discount=0.99, alpha=0.005).to(device)
    else:
        agent = DILO(qf=TwinQ(state_dim=obs_dim,act_dim=obs_dim),vf = ValueFunction(state_dim=obs_dim),policy=DeterministicPolicy(obs_dim,act_dim),
                                                        optimizer_factory=torch.optim.Adam,
                                                        lamda=FLAGS.lamda, maximizer=FLAGS.maximizer,beta=FLAGS.beta,ita=FLAGS.ita,tau=FLAGS.tau, gradient_type=FLAGS.grad,use_twinV=True, lr=3e-4, discount=0.99, alpha=0.005).to(device)

    trainer = TrainerDILO()


    best_eval_returns = -np.inf
    eval_returns = []
    for i in range(1, FLAGS.max_steps + 1): # Remove TQDM
        batch = dataset.sample(FLAGS.batch_size)
        expert_batch = expert_dataset.sample(FLAGS.batch_size)
        
        update_info, st = trainer.update(agent, batch, expert_batch)

        if i % FLAGS.eval_interval == 0:

            eval_stats = evaluate(agent.policy, env, FLAGS.eval_episodes, device,normalization_stats=normalization_stats)


            if eval_stats['return'] >= best_eval_returns:
                # Store best eval returns
                best_eval_returns = eval_stats['return']

            e_logger.log_tabular('Iterations', i)
            e_logger.log_tabular('AverageNormalizedReturn', eval_stats['return'])
            e_logger.log_tabular('SeenExpertV', update_info['expert_v_val'])
            e_logger.log_tabular('SeenReplayV', update_info['replay_v_val'])
            e_logger.log_tabular('UnseenExpertV', update_info['unseen_expert_v_val'])
            e_logger.log_tabular('UnseenReplayV', update_info['unseen_replay_v_val'])
            e_logger.log_tabular('UnseenExpertPolW', update_info['unseen_expert_pol_weight'])
            e_logger.log_tabular('UnseenReplayPolW', update_info['unseen_replay_pol_weight'])
            e_logger.log_tabular('Policy Loss', update_info['policy_loss'])
            e_logger.dump_tabular()
            eval_returns.append((i, eval_stats['return']))
            print("Iterations: {} Average Return: {}".format(i,eval_stats['return']))




if __name__ == '__main__':
    app.run(main)
