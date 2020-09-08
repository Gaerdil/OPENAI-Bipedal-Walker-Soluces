import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Critic(nn.Module):

	def __init__(self, state_dim, action_dim, init_var, seed):
		super(Critic, self).__init__()

		self.seed = torch.manual_seed(seed)


		self.state_dim = state_dim
		self.action_dim = action_dim
		self.init_var = init_var


		self.ls1 = nn.Linear(state_dim,512)
		self.cs = nn.Conv1d(1, 1, 9, stride= 2, padding=4)
		self.ls2 = nn.Linear(256,128)

		self.la = nn.Linear(action_dim,128)

		self.lsa = nn.Linear(256,128)

		self.lout = nn.Linear(128,1)




	def reinitWeights(self, reinit_deep = False):
		if reinit_deep: # /!\ worse in practice!
			self.ls1.weight.data = self.getInitWeightVar(self.ls1.weight.data.size())
			self.ls2.weight.data = self.getInitWeightVar(self.ls2.weight.data.size())
			self.la.weight.data = self.getInitWeightVar(self.la.weight.data.size())
			self.lsa.weight.data = self.getInitWeightVar(self.lsa.weight.data.size())
		self.lout.weight.data.uniform_(-self.init_var, self.init_var)

	def getInitWeightVar(self, size):  # Specific to actor model
		init_var = 1. / np.sqrt(size[0])
		return torch.Tensor(size).uniform_(-init_var, init_var)

	def forward(self, state, action):
		state = state.unsqueeze(1)
		if len(action.size()) == 2 :
			action = action.unsqueeze(1)
		s0 = F.relu(self.ls1(state))
		s1 = F.relu(self.cs(s0))
		s2 = F.relu(self.ls2(s1))
		a = F.relu(self.la(action))
		sa1 = torch.cat((s2,a),dim=2)

		sa2 = F.relu(self.lsa(sa1))
		val = self.lout(sa2)

		return val






