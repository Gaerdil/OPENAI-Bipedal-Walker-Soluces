from __future__ import division
from tqdm import *
import numpy as np
import torch

import matplotlib.pyplot as plt



class Main:
	def __init__(self,env,agent, num_epochs, max_steps, train = True, print_ = False, load_models = False, save_models = False):
		if load_models:
			agent.actor.load_state_dict(torch.load('checkpoint_actor.pth', map_location="cpu"))
			agent.critic.load_state_dict(torch.load('checkpoint_critic.pth', map_location="cpu"))
			agent.target_actor.load_state_dict(torch.load('checkpoint_actor_t.pth', map_location="cpu"))
			agent.target_critic.load_state_dict(torch.load('checkpoint_critic_t.pth', map_location="cpu"))

		best_reward = -1000
		self.Rewards = []
		self.Steps = []
		pbar = tqdm(range(num_epochs))
		for e in pbar:
			exploration = True
			if e % 5 == 0:
				exploration = False
			episode_rewards, episode_steps = episode(env, agent,  max_steps, exploration,train, print_)
			self.Rewards.append(episode_rewards)
			self.Steps.append(episode_steps)
			pbar.set_description("Episode reward: " + str(episode_rewards) + ", Survived steps: "+str(episode_steps)
								 +", Best reward: "+str(best_reward))
			if episode_rewards >= best_reward:
				best_reward = episode_rewards
				if save_models:
					torch.save(agent.actor.state_dict(), 'checkpoint_actor.pth')
					torch.save(agent.critic.state_dict(), 'checkpoint_critic.pth')
					torch.save(agent.target_actor.state_dict(), 'checkpoint_actor_t.pth')
					torch.save(agent.target_critic.state_dict(), 'checkpoint_critic_t.pth')



		print('End of Main experiment')

def episode(env, agent, max_steps, exploration, train,  print_):
	"""
	Represents a single episode
	:param env: the Gym environment
	:param agent: the learning agent from Agent class
	:param max_steps:
	:param exploration:
	:param train:
	:param print_:
	:return:
	"""
	episode_rewards = 0
	episode_steps = 0
	observation = env.reset()
	# print 'EPISODE :- ', _ep
	for step in range(max_steps):
		env.render()
		state = np.float32(observation)


		action = agent.getAction(state, exploration)


		new_observation, reward, done, info = env.step(action)

		episode_rewards += reward
		episode_steps += 1
		if step % 100 == 0 and print_:
			print(state, action, reward)

		# # dont update if this is validation
		# if _ep%50 == 0 or _ep>450:
		# 	continue

		if done:
			new_state = None
		else:
			new_state = np.float32(new_observation)
			agent.replayBuffer.add(state, action, reward, new_state)

		observation = new_observation

		# perform optimization
		if train : # and step > 5:
			agent.learn()


		if done:
			break
	return episode_rewards, episode_steps


def plotResults(rewards, steps,plot_frequency):
	plot_frequency = plot_frequency  # we wont plot everything, as it would make the graphs more difficult to read
	# And we will make a local average to keep track of the reward in a smoother way
	rewards, steps = np.array(rewards), np.array(steps) #converting to numpy array

	x_axis = [i for i in range(0, len(rewards), plot_frequency)]
	# We will average locally in the episode id the rewards and steps
	averaged_rewards = [np.mean(rewards[i:i + plot_frequency]) for i in range(0, len(rewards), plot_frequency)]
	averaged_steps = [np.mean(steps[i:i + plot_frequency]) for i in range(0, len(rewards), plot_frequency)]

	fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
	fig.suptitle("Evolution of the learning process of BipedalWalker ")

	axs[0].plot(x_axis, averaged_rewards)
	axs[0].set_xlabel("episode")
	axs[0].set_ylabel("reward")

	axs[1].plot(x_axis, averaged_steps)
	axs[1].set_xlabel("episode")
	axs[1].set_ylabel("survived steps")
