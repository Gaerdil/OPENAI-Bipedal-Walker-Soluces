import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim, init_var, seed):

		super(Actor, self).__init__()

		self.seed = torch.manual_seed(seed)

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.init_var = init_var

		self.l1 = nn.Linear(state_dim,256)
		self.cs = nn.Conv1d(1, 1, 9, stride=2, padding=4)
		self.l2 = nn.Linear(128,64)

		self.l3 = nn.Linear(64,32)

		self.lout = nn.Linear(32,action_dim)


	def reinitWeights(self, reinit_deep = False):
		if reinit_deep:
			self.l1.weight.data = self.getInitWeightVar(self.l1.weight.data.size())
			self.l2.weight.data = self.getInitWeightVar(self.l2.weight.data.size())
			self.l3.weight.data = self.getInitWeightVar(self.l3.weight.data.size())
		self.lout.weight.data.uniform_(-self.init_var, self.init_var)

	def getInitWeightVar(self, size):  # Specific to actor model
		init_var = 1. / np.sqrt(size[0])
		return torch.Tensor(size).uniform_(-init_var, init_var)

	def forward(self, state):
		if len(state.size()) == 1 : # not in training, 1 state only
			state = state.unsqueeze(0)
		state = state.unsqueeze(1)  # size()[0] is always the minibatch size
		s0 = F.relu(self.l1(state))
		s1 = F.relu(self.cs(s0))
		s2 = F.relu(self.l2(s1))
		s3 = F.relu(self.l3(s2))
		action = torch.tanh(self.lout(s3))

		action = action.squeeze(0) * self.action_lim

		return action



