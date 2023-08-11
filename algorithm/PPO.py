# pyright: reportUnboundVariable=false
# -*- encoding: utf-8 -*-
'''
@File : PPO.py
@Describe : Actor-critic network in PPO
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from algorithm.algo_config import device, GAMMA, EPS_CLIP, LR_ACTOR, LR_CRITIC, K_EPOCHS, LAMDA, ACTOR_STEP, CRITIC_STEP, ENTROPY_L, STATE_DIM

def orthogonal_init(layers, gain=1.0):
    for layer in layers:
        nn.init.orthogonal_(layer.weight, gain=gain)
        
        
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)
        orthogonal_init([self.fc1, self.fc2])
        orthogonal_init([self.alpha_head, self.beta_head], gain=0.01)
        
    def forward(self, s):
        a1 = torch.tanh(self.fc1(s))
        a = torch.tanh(self.fc2(a1))
        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0
        return alpha, beta
    
    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist
    
    def dist_mode(self, state):
        alpha, beta = self.forward(state)
        mode = (alpha) / (alpha + beta)
        return mode


class Critic(nn.Module):
    def __init__(self, state_dim, net_width=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.fc3 = nn.Linear(net_width, 1)
        self.activ1 = nn.ReLU()
        self.activ2 = nn.ReLU()
        orthogonal_init([self.fc1, self.fc2, self.fc3])
        
    def forward(self, s):
        s = self.activ1(self.fc1(s))
        s = self.activ2(self.fc2(s))
        v_s = self.fc3(s)
        return v_s
    
    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # actor
        self.actor = Actor(state_dim, action_dim)
        # critic
        self.critic = Critic(state_dim)
    
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            dist = self.actor.get_dist(state)
            action = dist.sample()
            # action = torch.clamp(action, 0, 1)
            action = torch.clamp(action, -1, 1)
            logprob_a = dist.log_prob(action).cpu().numpy().flatten()
        return action.cpu().numpy().flatten(), logprob_a
    
    def evaluate(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            dist = self.actor.get_dist(state)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)# action = self.actor(state)
        return action.cpu().numpy().flatten()
    
    
class PPOAgent:
    def __init__(self, args, as_dg_num, dgs):
        self.lamda = LAMDA
        self.gamma = GAMMA
        self.eps_clip = EPS_CLIP
        self.K_epochs = K_EPOCHS
        self.batch_size = args.batch_size
        self.s_dim = as_dg_num * STATE_DIM
        self.a_dim = as_dg_num
        self.data = []
        self.policy = ActorCritic(self.s_dim, self.a_dim).to(device)
        
        self.lr_a = LR_ACTOR
        self.lr_c = LR_CRITIC
        self.optimizer_actor = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr_a, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr_c, eps=1e-5)
        self.step_optim_actor = torch.optim.lr_scheduler.StepLR(self.optimizer_actor, step_size=ACTOR_STEP, gamma=0.95)
        self.step_optim_critic = torch.optim.lr_scheduler.StepLR(self.optimizer_critic, step_size=CRITIC_STEP, gamma=0.95)
        self.l2_reg = 1e-3
        self.update_time = 0
        self.dgs = dgs
        self.MseLoss = nn.MSELoss()
        
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        l = len(self.data)
        s_lst, a_lst, r_lst, s_prime_lst, logprob_a_lst, done_lst = np.zeros((l,self.s_dim)), np.zeros((l,self.a_dim)), np.zeros((l,1)), np.zeros((l,self.s_dim)), np.zeros((l,self.a_dim)), np.zeros((l,1))
        
        for i, transition in enumerate(self.data):
            s_lst[i], a_lst[i], r_lst[i], s_prime_lst[i], logprob_a_lst[i], done_lst[i] = transition
            
        self.data = [] # Clean history trajectory
        
        '''list to tensor'''
        with torch.no_grad():
            s, a, r, s_prime, logprob_a, done_mask = torch.tensor(s_lst, dtype=torch.float).to(device), \
                                                torch.tensor(a_lst, dtype=torch.float).to(device), \
                                                torch.tensor(r_lst, dtype=torch.float).to(device), \
                                                torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                                                torch.tensor(logprob_a_lst, dtype=torch.float).to(device), \
                                                torch.tensor(done_lst, dtype=torch.float).to(device)
            
        return s, a, r, s_prime, logprob_a, done_mask
    
    def update(self, writer):
        states, actions, rewards, next_states, logprob_a, dones = self.make_batch()
        # Monte Carlo estimate of returns
        adv = []
        gae = 0
        with torch.no_grad(): # adv and v_target have no gradient
            vs = self.policy.critic(states)
            vs_ = self.policy.critic(next_states)
            deltas = rewards + self.gamma * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(dones)):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(device)
            v_target = adv + vs
            # Normalizing the adv
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
            
        actor_loss_total, critic_loss_total = [], []
        optim_iter_num = int(math.ceil(states.shape[0] / self.batch_size))
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Shuffle the trajectory, Good for training
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            states, actions, v_target, adv, logprob_a = states[perm].clone(), actions[perm].clone(), v_target[perm].clone(), adv[perm].clone(),logprob_a[perm].clone()
            
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, states.shape[0]))
                distribution = self.policy.actor.get_dist(states[index])
                entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution.log_prob(actions[index])
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1, keepdim=True))
                
                # Finding Surrogate Loss
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * adv[index]
                
                # final loss of clipped objective PPO
                actor_loss = -torch.min(surr1, surr2) - ENTROPY_L * entropy
                actor_loss_total.append(actor_loss.mean().item())
                
                # Update
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(self.policy.actor.parameters(), 40)
                self.optimizer_actor.step()
                
                v_s = self.policy.critic(states[index])
                v_target_index = v_target[index]
                critic_loss = F.mse_loss(v_target_index, v_s)
                critic_loss_total.append(critic_loss.item())
                
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(self.policy.critic.parameters(), 0.5)
                self.optimizer_critic.step()
            # self.step_optim_actor.step()
            # self.step_optim_critic.step()
            
        writer.add_scalar('loss/actor_loss', np.mean(actor_loss_total), self.update_time)
        writer.add_scalar('loss/critic_loss', np.mean(critic_loss_total), self.update_time)
        # writer.add_scalar('lr/actor_lr', self.optimizer_actor.param_groups[0]['lr'], self.update_time)
        # writer.add_scalar('lr/critic_lr', self.optimizer_critic.param_groups[0]['lr'], self.update_time)
        self.update_time += 1
        
    def save(self, checkpoint_path):
        torch.save(self.policy.actor.state_dict(), checkpoint_path+"_actor.pth")
        torch.save(self.policy.critic.state_dict(), checkpoint_path+"_critic.pth")
        
    def load(self, checkpoint_path):
        self.policy.actor.load_state_dict(torch.load(checkpoint_path+"_actor.pth", map_location=lambda storage, loc: storage))
        self.policy.critic.load_state_dict(torch.load(checkpoint_path+"_critic.pth", map_location=lambda storage, loc: storage))
