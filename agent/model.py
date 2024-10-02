import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical

from agent.network import MLP

class DiscreteActorCritic(object):
    def __init__(self, config):
        self.config = config
        self.obs_dim, self.act_dim = self.config.obs_dim, self.config.act_dim
        self.hidden_dim = self.config.hidden_dim
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.actor = MLP(self.obs_dim, self.act_dim, self.hidden_dim)
        self.critic = MLP(self.obs_dim, 1, self.hidden_dim)

        self.actor_optim = Adam(self.actor.parameters(), lr = self.config.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.config.learning_rate)
        self.grad_enabled = False

    def enable_grad(self, mode):
        self.grad_enabled = mode
        
    def reset(self):
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

    def update(self):
        self.actor_optim.step()
        self.critic_optim.step()
    
    def get_policy(self, obs):
        with torch.set_grad_enabled(self.grad_enabled):
            outputs = self.actor(obs)
            probs = F.softmax(outputs, dim=-1)
            dist = Categorical(probs)
            return dist
    
    def get_action(self, obs):
        with torch.set_grad_enabled(self.grad_enabled):
            dist = self.get_policy(obs)
            act = dist.sample()
            return act
    
    def get_value(self, obs):
        with torch.set_grad_enabled(self.grad_enabled):
            value = self.critic(obs)
            return value
        
