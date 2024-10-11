import collections
from typing import Optional

import d4rl
import gym
import numpy as np
from tqdm import tqdm
import copy
import h5py
import wrappers


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

SNSBatch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'next_next_observations'])


MixedBatch = collections.namedtuple(
    'MixedBatch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations','is_expert'])


SNSMixedBatch = collections.namedtuple(
    'MixedBatch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'next_next_observations','is_expert'])



def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class SNSDataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray, next_next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.next_next_observations = next_next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return SNSBatch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx],
                     next_next_observations=self.next_next_observations[indx])



class MixedDataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray, is_expert: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.is_expert = is_expert
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return MixedBatch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx],
                     is_expert=self.is_expert[indx])


class SNSMixedDataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,next_next_observations: np.ndarray, is_expert: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.next_next_observations = next_next_observations
        self.is_expert = is_expert
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return SNSMixedBatch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx],
                     next_next_observations=self.next_next_observations[indx],
                     is_expert=self.is_expert[indx])

class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 transitions= None,
                 env_name=""):

        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        # if transitions is not None and "antmaze" in env_name:
        dones_float[-1] = 1
        if transitions is not None:
            cropped_dataset = {}
            cropped_dataset['observations'] = dataset['observations'][:transitions]
            cropped_dataset['actions'] = dataset['actions'][:transitions]
            cropped_dataset['rewards'] = dataset['rewards'][:transitions]
            cropped_dataset['terminals'] = dataset['terminals'][:transitions]
            cropped_dataset['next_observations'] = dataset['next_observations'][:transitions]
            super().__init__(cropped_dataset['observations'].astype(np.float32),
                            actions=cropped_dataset['actions'].astype(np.float32),
                            rewards=cropped_dataset['rewards'].astype(np.float32),
                            masks=1.0 - cropped_dataset['terminals'].astype(np.float32),
                            dones_float=dones_float[:transitions].astype(np.float32),
                            next_observations=cropped_dataset['next_observations'].astype(
                                np.float32),
                            size=transitions-1)
        else:    
            super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))




class SNSD4RLDataset(SNSDataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 transitions= None,
                 env_name=""):

        dataset = d4rl.qlearning_dataset(env)
        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])
        mod_dataset = {
            'observations':np.zeros_like(dataset['observations']),
            'actions':np.zeros_like(dataset['actions']),
            'rewards':np.zeros_like(dataset['rewards']),
            'terminals':np.zeros_like(dataset['terminals']),
            'next_observations':np.zeros_like(dataset['next_observations']),
            'next_next_observations':np.zeros_like(dataset['next_observations'])
        }
        mod_dataset_size = 0
        for i in range(len(dones_float) - 1):
            
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                mod_dataset['observations'][mod_dataset_size] = dataset['observations'][i]
                mod_dataset['actions'][mod_dataset_size] = dataset['actions'][i]
                mod_dataset['rewards'][mod_dataset_size] = dataset['rewards'][i]
                mod_dataset['terminals'][mod_dataset_size] = dataset['terminals'][i]
                mod_dataset['next_observations'][mod_dataset_size] = dataset['next_observations'][i]
                mod_dataset['next_next_observations'][mod_dataset_size] = dataset['next_observations'][i+1]
                mod_dataset_size+=1
                dones_float[i] = 0
        # if transitions is not None and "antmaze" in env_name:
        dones_float[-1] = 1
        if transitions is not None:
            cropped_dataset = {}
            cropped_dataset['observations'] = mod_dataset['observations'][:transitions]
            cropped_dataset['actions'] = mod_dataset['actions'][:transitions]
            cropped_dataset['rewards'] = mod_dataset['rewards'][:transitions]
            cropped_dataset['terminals'] = mod_dataset['terminals'][:transitions]
            cropped_dataset['next_observations'] = mod_dataset['next_observations'][:transitions]
            cropped_dataset['next_next_observations'] = mod_dataset['next_next_observations'][:transitions]


            super().__init__(cropped_dataset['observations'].astype(np.float32),
                            actions=cropped_dataset['actions'].astype(np.float32),
                            rewards=cropped_dataset['rewards'].astype(np.float32),
                            masks=1.0 - cropped_dataset['terminals'].astype(np.float32),
                            dones_float=cropped_dataset['terminals'].astype(np.float32),
                            next_observations=cropped_dataset['next_observations'].astype(
                                np.float32),
                            next_next_observations=cropped_dataset['next_next_observations'].astype(
                                np.float32),
                            size=transitions-1)
        else:    
            super().__init__(mod_dataset['observations'].astype(np.float32),
                         actions=mod_dataset['actions'].astype(np.float32),
                         rewards=mod_dataset['rewards'].astype(np.float32),
                         masks=1.0 - mod_dataset['terminals'].astype(np.float32),
                         dones_float=mod_dataset['terminals'].astype(np.float32),
                         next_observations=mod_dataset['next_observations'].astype(
                             np.float32),
                        next_next_observations=mod_dataset['next_next_observations'].astype(
                             np.float32),
                         size=mod_dataset_size)

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def get_dataset(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    return data_dict

class D4RLMixedDataset(MixedDataset):
    def __init__(self,
                 env: gym.Env,
                 expert_env: gym.Env,
                 expert_trajectories: int,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 transitions= None,
                 env_name=""):

        if 'kitchen' in env_name:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 280
        elif 'pen' in env_name:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 100
        elif 'door' in env_name or 'hammer' in env_name or 'relocate' in env_name:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 200    
        else:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 1000
        
        if 'Grid' in  env_name:
            expert_dataset = {'observations':[],'actions':[],'rewards':[],'terminals':[],'next_observations':[]}
            expert_transitions = 0
            for traj_id in range(1):
                env.reset()
                goal_state = env.goal
                state = np.array([-1.2,-1.2])
                env.set_state(state)
                for i in range(5):
                    act = np.clip(goal_state-state,-0.25,0.25)
                    next_state, rew, done, _ = env.step(act)
                    expert_dataset['observations'].append(state.reshape(-1))
                    expert_dataset['actions'].append(act.reshape(-1))
                    expert_dataset['rewards'].append(rew)
                    expert_dataset['terminals'].append(done)
                    expert_dataset['next_observations'].append(next_state.reshape(-1))
                    expert_transitions+=1
                    state = next_state
            expert_dataset['observations'] = np.array(expert_dataset['observations'])
            expert_dataset['actions'] = np.array(expert_dataset['actions'])
            expert_dataset['rewards'] = np.array(expert_dataset['rewards'])
            expert_dataset['terminals'] = np.array(expert_dataset['terminals'])
            expert_dataset['next_observations'] = np.array(expert_dataset['next_observations'])

        else:
            expert_dataset = d4rl.qlearning_dataset(expert_env)

            expert_transitions = 0
            traj_count= 0
            episode_step=0
            

            for i in range(expert_dataset['observations'].shape[0]):
                episode_step+=1
                
                if episode_step == max_steps or expert_dataset['terminals'][i]:
                    # if episode_step == max_steps:
                    traj_count+=1
                    episode_step = 0
                if traj_count == expert_trajectories:
                    expert_transitions = i
                    break

        combined_dataset = copy.copy(dataset)
        combined_dataset['observations'] = np.concatenate((dataset['observations'], expert_dataset['observations'][:expert_transitions]),axis=0)
        combined_dataset['actions'] = np.concatenate((dataset['actions'], expert_dataset['actions'][:expert_transitions]),axis=0)
        combined_dataset['rewards'] = np.concatenate((dataset['rewards'], expert_dataset['rewards'][:expert_transitions]),axis=0)
        combined_dataset['terminals'] = np.concatenate((dataset['terminals'], expert_dataset['terminals'][:expert_transitions]),axis=0)
        combined_dataset['next_observations'] = np.concatenate((dataset['next_observations'], expert_dataset['next_observations'][:expert_transitions]),axis=0)
        combined_dataset['is_expert'] = np.concatenate((np.zeros(dataset['observations'].shape[0]), np.ones(expert_transitions)),axis=0)

        dones_float = np.zeros_like(combined_dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(combined_dataset['observations'][i + 1] -
                              combined_dataset['next_observations'][i]
                              ) > 1e-6 or combined_dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1
        if transitions is not None:
            super().__init__(combined_dataset['observations'][:transitions].astype(np.float32),
                         actions=combined_dataset['actions'][:transitions].astype(np.float32),
                         rewards=combined_dataset['rewards'][:transitions].astype(np.float32),
                         masks=1.0 - combined_dataset['terminals'][:transitions].astype(np.float32),
                         dones_float=dones_float[:transitions].astype(np.float32),
                         next_observations=combined_dataset['next_observations'][:transitions].astype(
                             np.float32),
                        is_expert = combined_dataset['is_expert'][:transitions].astype(np.float32),
                         size=transitions)
        else:    
            super().__init__(combined_dataset['observations'].astype(np.float32),
                         actions=combined_dataset['actions'].astype(np.float32),
                         rewards=combined_dataset['rewards'].astype(np.float32),
                         masks=1.0 - combined_dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=combined_dataset['next_observations'].astype(
                             np.float32),
                        is_expert=combined_dataset['is_expert'].astype(np.float32),
                         size=len(combined_dataset['observations']))



class SNSD4RLMixedDataset(SNSMixedDataset):
    def __init__(self,
                 env: gym.Env,
                 expert_env: gym.Env,
                 expert_trajectories: int,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 transitions= None,
                 env_name=""):
        if 'kitchen' in env_name:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 280
        elif 'pen' in env_name:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 100
        elif 'door' in env_name or 'hammer' in env_name or 'relocate' in env_name:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 200    
        else:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 1000
        
        if 'Grid' in  env_name:
            expert_dataset = {'observations':[],'actions':[],'rewards':[],'terminals':[],'next_observations':[]}
            expert_transitions = 0
            for traj_id in range(1):
                env.reset()
                goal_state = env.goal
                state = np.array([-1.2,-1.2])
                env.set_state(state)
                for i in range(5):
                    act = np.clip(goal_state-state,-0.25,0.25)
                    next_state, rew, done, _ = env.step(act)
                    expert_dataset['observations'].append(state.reshape(-1))
                    expert_dataset['actions'].append(act.reshape(-1))
                    expert_dataset['rewards'].append(rew)
                    expert_dataset['terminals'].append(done)
                    expert_dataset['next_observations'].append(next_state.reshape(-1))
                    expert_transitions+=1
                    state = next_state
            expert_dataset['observations'] = np.array(expert_dataset['observations'])
            expert_dataset['actions'] = np.array(expert_dataset['actions'])
            expert_dataset['rewards'] = np.array(expert_dataset['rewards'])
            expert_dataset['terminals'] = np.array(expert_dataset['terminals'])
            expert_dataset['next_observations'] = np.array(expert_dataset['next_observations'])

        else:
            expert_dataset = d4rl.qlearning_dataset(expert_env)

            expert_transitions = 0
            traj_count= 0
            episode_step=0
            

            for i in range(expert_dataset['observations'].shape[0]):
                episode_step+=1
                
                if episode_step == max_steps or expert_dataset['terminals'][i]:
                    # if episode_step == max_steps:
                    traj_count+=1
                    episode_step = 0
                if traj_count == expert_trajectories:
                    expert_transitions = i
                    break
        combined_dataset = copy.copy(dataset)
        combined_dataset['observations'] = np.concatenate((dataset['observations'], expert_dataset['observations'][:expert_transitions]),axis=0)
        combined_dataset['actions'] = np.concatenate((dataset['actions'], expert_dataset['actions'][:expert_transitions]),axis=0)
        combined_dataset['rewards'] = np.concatenate((dataset['rewards'], expert_dataset['rewards'][:expert_transitions]),axis=0)
        combined_dataset['terminals'] = np.concatenate((dataset['terminals'], expert_dataset['terminals'][:expert_transitions]),axis=0)
        combined_dataset['next_observations'] = np.concatenate((dataset['next_observations'], expert_dataset['next_observations'][:expert_transitions]),axis=0)
        combined_dataset['is_expert'] = np.concatenate((np.zeros(dataset['observations'].shape[0]), np.ones(expert_transitions)),axis=0)

        dones_float = np.zeros_like(combined_dataset['rewards'])
        mod_dataset = {
            'observations':np.zeros_like(combined_dataset['observations']),
            'actions':np.zeros_like(combined_dataset['actions']),
            'rewards':np.zeros_like(combined_dataset['rewards']),
            'terminals':np.zeros_like(combined_dataset['terminals']),
            'next_observations':np.zeros_like(combined_dataset['next_observations']),
            'next_next_observations':np.zeros_like(combined_dataset['next_observations']),
            'is_expert':np.zeros_like(combined_dataset['is_expert'])
        }
        mod_dataset_size = 0
        for i in range(len(dones_float) - 1):
            
            if np.linalg.norm(combined_dataset['observations'][i + 1] -
                              combined_dataset['next_observations'][i]
                              ) > 1e-6 or combined_dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                mod_dataset['observations'][mod_dataset_size] = combined_dataset['observations'][i]
                mod_dataset['actions'][mod_dataset_size] = combined_dataset['actions'][i]
                mod_dataset['rewards'][mod_dataset_size] = combined_dataset['rewards'][i]
                mod_dataset['terminals'][mod_dataset_size] = combined_dataset['terminals'][i]
                mod_dataset['next_observations'][mod_dataset_size] = combined_dataset['next_observations'][i]
                mod_dataset['next_next_observations'][mod_dataset_size] = combined_dataset['next_observations'][i+1]
                mod_dataset['is_expert'][mod_dataset_size] = combined_dataset['is_expert'][i]
                mod_dataset_size+=1
                dones_float[i] = 0


        dones_float[-1] = 1
        if transitions is not None:
            super().__init__(mod_dataset['observations'][:transitions].astype(np.float32),
                         actions=mod_dataset['actions'][:transitions].astype(np.float32),
                         rewards=mod_dataset['rewards'][:transitions].astype(np.float32),
                         masks=1.0 - mod_dataset['terminals'][:transitions].astype(np.float32),
                         dones_float=mod_dataset['terminals'][:transitions].astype(np.float32),
                         next_observations=mod_dataset['next_observations'][:transitions].astype(
                             np.float32),
                         next_next_observations=mod_dataset['next_next_observations'][:transitions].astype(
                             np.float32),
                        is_expert = mod_dataset['is_expert'][:transitions].astype(np.float32),
                         size=transitions)
        else:    
            super().__init__(mod_dataset['observations'].astype(np.float32),
                         actions=mod_dataset['actions'].astype(np.float32),
                         rewards=mod_dataset['rewards'].astype(np.float32),
                         masks=1.0 - mod_dataset['terminals'].astype(np.float32),
                         dones_float= mod_dataset['terminals'].astype(np.float32),
                         next_observations=mod_dataset['next_observations'].astype(
                             np.float32),
                        next_next_observations=mod_dataset['next_next_observations'].astype(
                             np.float32),
                        is_expert=mod_dataset['is_expert'].astype(np.float32),
                         size=mod_dataset_size)

class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


def make_env_and_dataset(env_name: str,
                         seed: int, expert_trajectories:int,normalize_obs=False):
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    expert_dataset = None
    
    if 'kitchen' in env_name:
        expert_env = gym.make(f"kitchen-complete-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=183)
    elif 'halfcheetah-random' in env_name:
        expert_env = gym.make(f"halfcheetah-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=1000)
    elif 'halfcheetah-medium' in env_name:
        expert_env = gym.make(f"halfcheetah-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=1000)
    elif 'hopper-random' in env_name:
        expert_env = gym.make(f"hopper-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=1000)
    elif "walker2d-random" in env_name:
        expert_env = gym.make(f"walker2d-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=1000)
    elif "ant-random" in env_name:
        expert_env = gym.make(f"ant-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=1000)
    elif "ant-medium" in env_name:
        expert_env = gym.make(f"ant-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=1000)
    elif "door-human" in env_name:
        expert_env = gym.make(f"door-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=200)
    elif "door-cloned" in env_name:
        expert_env = gym.make(f"door-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=200)
    elif "hammer-human" in env_name:
        expert_env = gym.make(f"hammer-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=200)
    elif "hammer-cloned" in env_name:
        expert_env = gym.make(f"hammer-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = SNSD4RLDataset(expert_env, transitions=200)


    offline_min=None
    offline_max=None

    if 'kitchen' not in env_name:
        dataset = SNSD4RLMixedDataset(env, expert_env, expert_trajectories=expert_trajectories,env_name=env_name)
    else:
        
        dataset = SNSD4RLMixedDataset(env, expert_env, expert_trajectories=1,env_name=env_name) #D4RLDataset(env)
    print("Expert dataset size: {} Offline dataset size: {}".format(expert_dataset.observations.shape[0],dataset.observations.shape[0]))

    normalization_stats  = {}
    if normalize_obs:
        obs_mean = np.concatenate([expert_dataset.observations, dataset.observations]).mean(axis=0)
        obs_std = np.concatenate([expert_dataset.observations, dataset.observations]).std(axis=0)+1e-3
        dataset.observations = (dataset.observations-obs_mean)/obs_std
        expert_dataset.observations = (expert_dataset.observations-obs_mean)/obs_std
        dataset.next_observations = (dataset.next_observations-obs_mean)/obs_std
        expert_dataset.next_observations = (expert_dataset.next_observations-obs_mean)/obs_std
        normalization_stats['obs_mean'] = obs_mean
        normalization_stats['obs_std'] = obs_std

    return env, dataset, expert_dataset, offline_min, offline_max, normalization_stats

