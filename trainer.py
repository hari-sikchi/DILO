import time
import torch 


class Trainer():
    def __init__(self):
        pass

    def update(self, model, batch, expert_batch, eval=False):
        t0 = time.time()
        metrics = dict()


        t1 = time.time()

        ## Batch
        obs, acts,next_obs = torch.FloatTensor(batch.observations), torch.FloatTensor(batch.actions), torch.FloatTensor(batch.next_observations)
        expert_obs, expert_acts, expert_next_obs = torch.FloatTensor(expert_batch.observations), torch.FloatTensor(expert_batch.actions), torch.FloatTensor(expert_batch.next_observations)
        terminals = 1-torch.FloatTensor(batch.masks)
        expert_terminals = 1-torch.FloatTensor(expert_batch.masks)
        is_expert = torch.FloatTensor(batch.is_expert)
        metrics =  model.update(obs.float().cuda(),acts.float().cuda(), next_obs.float().cuda(), terminals.float().cuda(),is_expert.float().cuda(),expert_obs.float().cuda(),expert_acts.float().cuda(), expert_next_obs.float().cuda(),expert_terminals.float().cuda() )

        t2 = time.time()

        return metrics, f"Load time {t1-t0}, Batch time {t2-t1}, Update time {t2-t1}, V Loss {metrics['v_loss']}"


class TrainerSNS():
    def __init__(self):
        pass

    def update(self, model, batch, expert_batch, eval=False):
        t0 = time.time()
        metrics = dict()


        t1 = time.time()

        ## Batch
        obs, acts,next_obs, next_next_obs = torch.FloatTensor(batch.observations), torch.FloatTensor(batch.actions), torch.FloatTensor(batch.next_observations), torch.FloatTensor(batch.next_next_observations)
        expert_obs, expert_acts, expert_next_obs, expert_next_next_obs = torch.FloatTensor(expert_batch.observations), torch.FloatTensor(expert_batch.actions), torch.FloatTensor(expert_batch.next_observations), torch.FloatTensor(expert_batch.next_next_observations)
        terminals = 1-torch.FloatTensor(batch.masks)
        expert_terminals = 1-torch.FloatTensor(expert_batch.masks)
        is_expert = torch.FloatTensor(batch.is_expert)
        metrics =  model.update(obs.float().cuda(),acts.float().cuda(), next_obs.float().cuda(),next_next_obs.float().cuda(), terminals.float().cuda(),is_expert.float().cuda(),expert_obs.float().cuda(),expert_acts.float().cuda(), expert_next_obs.float().cuda(),expert_next_next_obs.float().cuda(),expert_terminals.float().cuda() )


        t2 = time.time()

        return metrics, f"Load time {t1-t0}, Batch time {t2-t1}, Update time {t2-t1}, V Loss {metrics['v_loss']}"
