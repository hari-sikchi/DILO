import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from dilo_utils import DEFAULT_DEVICE, update_exponential_moving_average


EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def f_prime_inverse(residual, name='Pearson_chi_square', temperatrue=3.0):
    if name == "Reverse_KL":
        return torch.exp(residual * temperatrue)
    elif name == "Pearson_chi_square":
        return torch.max(residual, torch.zeros_like(residual))



class DILO(nn.Module):
    def __init__(self,  qf,vf,policy, optimizer_factory,
                 lamda, maximizer, beta,ita,tau, gradient_type,use_twinV=False, lr=3e-4, discount=0.99, alpha=0.005, max_steps=int(1e6)):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.v_target = copy.deepcopy(vf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters(), lr = lr)
        self.q_optimizer = optimizer_factory(self.qf.parameters(), lr = lr)
        self.policy_optimizer = optimizer_factory(self.policy.parameters(), lr = 1e-4)
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.lamda = lamda # 0.8
        self.maximizer = maximizer
        self.beta = beta # 0.5
        self.ita = ita # 0.5
        self.tau = tau # 3.0/50
        self.discount = discount
        self.alpha = alpha
        self.gradient_type = gradient_type
        self.update_steps = 0
        self.use_twinV = use_twinV

    def f_star(self, residual, type='chi_square'):
        if type=='chi_square':
            omega_star = torch.max(residual / 2 + 1, torch.zeros_like(residual))
            return residual * omega_star - (omega_star - 1)**2
        else:
            raise NotImplementedError("f star for divergence not implemented")


    def update_full(self, obs,acts,next_obs,next_next_obs,terminals, is_expert, expert_obs, expert_acts, expert_next_obs, expert_next_next_obs, expert_terminals):
        v_loss_val = 0.0
        metrics = {}

        if self.use_twinV:
            v_curr =  self.qf.both(obs, next_obs)
            v_next =  self.qf.both(next_obs, next_next_obs)
            gt_v_curr = self.qf.both(expert_obs, expert_next_obs)
            gt_v_next = self.qf.both(expert_next_obs, expert_next_next_obs)
            v_curr = torch.stack(v_curr,dim=1)
            v_next = torch.stack(v_next,dim=1)
            gt_v_curr = torch.stack(gt_v_curr,dim=1)
            gt_v_next = torch.stack(gt_v_next,dim=1)
        else:
            v_curr =  self.qf(obs)
            v_next =  self.qf(next_obs)
            gt_v_curr = self.qf(expert_obs)

        v_curr_target = self.q_target(obs, next_obs).detach()
        v_next_target = self.q_target(next_obs,next_next_obs).detach()
        

        # Update value function
        if self.maximizer == 'smoothed_chi':
            if self.use_twinV:
                gt_v_next = self.qf.both(expert_next_obs, expert_next_next_obs)
                gt_v_next = torch.stack(gt_v_next,dim=1)
                backward_residual = ((1. - terminals.view(-1,1).float()) * self.discount * v_next   - v_curr_target.view(-1,1))
                forward_residual = ((1. - terminals.view(-1,1).float()) * self.discount * v_next_target.view(-1,1)  - v_curr)
                gt_v_curr_target = self.q_target(expert_obs,expert_next_obs).detach()
                gt_v_next_target = self.q_target(expert_next_obs,expert_next_next_obs).detach()
                gt_backward_residual = (1. - expert_terminals.view(-1,1).float()) * self.discount * gt_v_next - gt_v_curr_target.view(-1,1)
                gt_forward_residual =  (1. - expert_terminals.view(-1,1).float()) * self.discount * gt_v_next_target.view(-1,1) - gt_v_curr
            else:
                gt_v_next = self.qf(expert_next_obs,expert_next_next_obs)
                backward_residual = ((1. - terminals.float()) * self.discount * v_next   - v_curr_target)
                forward_residual = ((1. - terminals.float()) * self.discount * v_next_target  - v_curr)
                gt_v_curr_target = self.q_target(expert_obs,expert_next_obs).detach()
                gt_v_next_target = self.q_target(expert_next_obs,expert_next_next_obs).detach()
                gt_backward_residual = (1. - expert_terminals.float()) * self.discount * gt_v_next - gt_v_curr_target
                gt_forward_residual =  (1. - expert_terminals.float()) * self.discount * gt_v_next_target - gt_v_curr
            
            backward_dual_loss = (self.lamda)*self.ita*(self.beta *  self.f_star(gt_backward_residual) + (1-self.beta) * self.f_star(backward_residual) - (1-self.beta)*backward_residual).mean() # First iteartion that worked decently
            forward_dual_loss = (self.lamda)* (self.beta * self.f_star(gt_forward_residual) + (1-self.beta) * self.f_star(forward_residual) - (1-self.beta)*forward_residual).mean() # First iteartion that worked decently
        else:
            raise NotImplementedError('Unavailable divergence for full gradient update')
        # For logging
        expert_v_val = gt_v_curr.mean().item()
        replay_v_val = v_curr.mean().item()
        if is_expert.sum()>0:
            unseen_expert_v_val = gt_v_curr[is_expert.bool()].mean().item()
        else:
            unseen_expert_v_val = -1
        if (1-is_expert).sum()>0:
            unseen_replay_v_val = v_curr[(1-is_expert).bool()].mean().item()
        else:
            unseen_replay_v_val = -1
        if self.use_twinV:
            pi_residual = forward_residual[:,0].clone().detach()
        else:
            pi_residual = forward_residual.clone().detach()

        v_loss_val += 0.5*(forward_dual_loss.item() + backward_dual_loss.item())
        self.q_optimizer.zero_grad(set_to_none=True)
        forward_grad_list, backward_grad_list = [], []
        forward_dual_loss.backward(retain_graph=True)
        for param in list(self.qf.parameters()):
            forward_grad_list.append(param.grad.clone().detach().reshape(-1))
        backward_dual_loss.backward()
        for i, param in enumerate(list(self.qf.parameters())):
            backward_grad_list.append(param.grad.clone().detach().reshape(-1) - forward_grad_list[i])
        forward_grad, backward_grad = torch.cat(forward_grad_list), torch.cat(backward_grad_list)
        parallel_coef = (torch.dot(forward_grad, backward_grad) / max(torch.dot(forward_grad, forward_grad),
                                                                      1e-10)).item()  # avoid zero grad caused by f*
        forward_grad = (1 - parallel_coef) * forward_grad + backward_grad
        
        param_idx = 0
        for i, grad in enumerate(forward_grad_list):
            forward_grad_list[i] = forward_grad[param_idx: param_idx + grad.shape[0]]
            param_idx += grad.shape[0]
        # reset gradient and calculate
        self.q_optimizer.zero_grad(set_to_none=True)
        if self.maximizer == 'smoothed_chi':
            v_loss = (1-self.lamda)*v_curr.mean()
        else:
            v_loss = (gt_v_curr-200).pow(2).mean()
        v_loss.backward()
        for i, param in enumerate(list(self.qf.parameters())):
            param.grad += forward_grad_list[i].reshape(param.grad.shape)

        self.q_optimizer.step()

        v_loss_val += v_loss.item()
        
        # Update policy network
        temperature = self.tau
        pi_residual=self.qf(obs, next_obs).detach()
        weight = torch.exp(temperature*pi_residual)
        weight = torch.clamp_max(weight, EXP_ADV_MAX).detach()
        policy_out = self.policy(obs)
        bc_losses = (acts-policy_out).pow(2).sum(1)
        # bc_losses = -policy_out.log_prob(acts)
        policy_loss = torch.mean(weight * bc_losses)
        
        if is_expert.sum()>0:
            unseen_expert_pol_weight = weight[is_expert.bool()].mean().item()
        else:
            unseen_expert_pol_weight = -1
        if (1-is_expert).sum()>0:
            unseen_replay_pol_weight = weight[(1-is_expert).bool()].mean().item()
        else:
            unseen_replay_pol_weight = -1
        
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()
        
        # # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        metrics['v_loss'] = v_loss_val
        metrics['policy_loss'] = policy_loss.item()
        metrics['expert_v_val'] = expert_v_val
        metrics['replay_v_val'] = replay_v_val
        metrics['unseen_expert_v_val'] = unseen_expert_v_val
        metrics['unseen_replay_v_val'] = unseen_replay_v_val
        metrics['unseen_expert_pol_weight'] = unseen_expert_pol_weight
        metrics['unseen_replay_pol_weight'] = unseen_replay_pol_weight
        return metrics


    def update(self, obs,acts,next_obs,next_next_obs, terminals, is_expert, expert_obs, expert_acts, expert_next_obs, expert_next_next_obs, expert_terminals):
        if self.gradient_type == 'semi':
            return self.update_semi( obs,acts,next_obs,next_next_obs,terminals, is_expert, expert_obs, expert_acts, expert_next_obs, expert_next_next_obs, expert_terminals)
        elif self.gradient_type == 'full':
            return self.update_full(obs,acts,next_obs,next_next_obs,terminals, is_expert, expert_obs, expert_acts, expert_next_obs, expert_next_next_obs, expert_terminals)
        