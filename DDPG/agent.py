from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import utils
from Critic import *
from  Actor import *




class Agent:

	def __init__(self, state_dim, action_dim, action_lim, lrs, taus, gamma, init_var, weight_decays, batch_size, max_memory_size, seed):

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.gamma = gamma
		self.taus = taus
		self.lrs = lrs
		self.batch_size = batch_size
		self.replayBuffer = utils.MemoryBuffer(max_memory_size, self.batch_size)
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckProcess(self.action_dim, mu = 0, theta = 0.15, sigma = 0.2)

		self.actor = Actor(self.state_dim, self.action_dim, self.action_lim, init_var, seed +1)
		self.target_actor = Actor(self.state_dim, self.action_dim, self.action_lim,init_var, seed +1)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lrs[0], weight_decay= weight_decays[0])

		self.critic = Critic(self.state_dim, self.action_dim, init_var, seed)
		self.target_critic = Critic(self.state_dim, self.action_dim,init_var, seed)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lrs[1], weight_decay= weight_decays[1])

		self.reinit()


	def reinit(self):
		self.actor.reinitWeights()
		self.critic.reinitWeights()
		utils.copyModel(self.target_actor, self.actor)
		utils.copyModel(self.target_critic, self.critic)


	def getAction(self, state, exploration = False):
		state = Variable(torch.from_numpy(state))
		#self.actor.eval() # Torch eval mode (if using dropout or batchnorm layers)
		action = self.actor.forward(state).detach().data.numpy()[0]
		if exploration:
			action = action + (self.noise.sample() * self.action_lim)
		#self.actor.train() # Torch train mode (if using dropout or batchnorm layers), not real TD learning update
		return action


	def learn(self):

		states,actions,rewards, next_states = self.replayBuffer.sample()

		self.optimizeCritic(states, actions, rewards, next_states)
		self.optimizeActor(states)

		utils.softModelUpdate(self.target_actor, self.actor, self.taus[0])
		utils.softModelUpdate(self.target_critic, self.critic, self.taus[1])

	def optimizeCritic(self, state,action,reward,next_state):

		# Use target actor exploitation policy here for loss evaluation
		next_action = self.target_actor.forward(next_state).detach() # prevision of next actions by target actor
		next_val = torch.squeeze(self.target_critic.forward(next_state, next_action).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
		y_expected = reward + self.gamma * next_val  # NO DONE?
		# y_pred = Q( s1, a1)
		y_predicted = torch.squeeze(self.critic.forward(state, action))
		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

	def optimizeActor(self, state):

		action_pred = self.actor.forward(state)
		loss_actor = -1 * torch.sum(self.critic.forward(state, action_pred))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()


