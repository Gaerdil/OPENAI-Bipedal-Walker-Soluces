import numpy as np
import torch
from collections import deque
from torch.autograd import Variable

np_seed = np.random.seed(0)


def copyModel(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)



def softModelUpdate(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)



class OrnsteinUhlenbeckProcess:

	def __init__(self, action_dim, mu , theta , sigma ):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.x = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.x = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.x)
		dx = dx + self.sigma * np.random.randn(len(self.x))
		self.x = self.x + dx
		return self.x



class MemoryBuffer:

	def __init__(self, max_size, batch_size):
		self.buffer = deque()
		self.max_size = max_size
		self.batch_size = batch_size
		self.size = 0

	def len(self):
		return self.size

	def add(self, state, action, reward, next_state):
		self.size += 1
		if self.size > self.max_size:
			self.size = self.max_size
			self.buffer.popleft()  # most ancient experiments : gotten rid of
		self.buffer.append((state, action, reward, next_state))

	def sample(self):
		current_batch_size = min(self.batch_size, self.size)
		indices = np.random.randint(0, self.size, current_batch_size)
		mini_batch = [self.buffer[i] for i in indices]

		states_tensor = Variable(torch.from_numpy(np.float32([e[0] for e in mini_batch])))
		actions_tensor = Variable(torch.from_numpy(np.float32([e[1] for e in mini_batch])))
		rewards_tensor = Variable(torch.from_numpy(np.float32([e[2] for e in mini_batch])))
		next_states_tensor = Variable(torch.from_numpy(np.float32([e[3] for e in mini_batch])))

		return states_tensor, actions_tensor, rewards_tensor, next_states_tensor


