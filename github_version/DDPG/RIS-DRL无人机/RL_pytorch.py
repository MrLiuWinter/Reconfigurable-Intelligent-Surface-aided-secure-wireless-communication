import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
np.random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

class Net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_output):
		super(Net, self).__init__()
		self.el = nn.Linear(n_feature, n_hidden)
		self.q = nn.Linear(n_hidden, n_output)

	def forward(self, x):
		x = self.el(x)
		x = F.relu(x)
		x = self.q(x)
		return x

class DoubleDQN():
	def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.005, reward_decay=0.9, e_greedy=0.9,    #初始参数
				replace_target_iter=200, memory_size=3200, batch_size=32, e_greedy_increment=None, double_q=True, ris=True, passive_shift =True):
		self.n_actions = n_actions                           #行为#
		self.n_hidden = n_hidden                             #网络隐藏层
		self.n_features = n_features                         #状态
		self.lr = learning_rate                              #学习速率
		self.gamma = reward_decay                            #汇报折扣率
		self.epsilon_max = e_greedy                          #贪婪度
		self.replace_target_iter = replace_target_iter       #target更新次数
		self.memory_size = memory_size                       #经验池大小
		self.batch_size = batch_size                         #神经网络批训练
		self.epsilon_increment = e_greedy_increment          #贪婪
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
		self.double_q = double_q
		self.learn_step_counter = 0                         #学习步骤
		self.memory = np.zeros((self.memory_size, n_features*2+3+1+1+1))   #feature是状态有两个时刻+行为+奖励
		self._build_net()
		self.cost_his = []                                  #存放数据
		self.ris=ris
		self.passive_shift = passive_shift  #11


	def _build_net(self):
		self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
		self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
		self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)
		self.loss_func = nn.MSELoss()

	def store_transition(self, s, a, r, s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((s, a, r, s_))
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter += 1

	def choose_action(self, observation):
		observation = torch.Tensor(observation[np.newaxis, :])  #数据转化
		actions_value = self.q_eval(observation)                #评估网络，获取Q只
		action = torch.max(actions_value, dim=1)[1]             #根据Q值最大的获取行动action
		if not hasattr(self, 'q'):
			self.q = []
			self.running_q = 0
		self.running_q = self.running_q*0.99 + 0.01 * torch.max(actions_value, dim=1)[0]
		self.q.append(self.running_q)

		if np.random.uniform() > self.epsilon:                  #贪婪度进行探索随机选取
			action = np.random.randint(0, self.n_actions)
		return action

	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.q_target.load_state_dict(self.q_eval.state_dict())       # 多少步后，对qtarget进行更新

		if self.memory_counter > self.memory_size:                       #经验池进行存储，到达一定数量开始循环存储
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:                                                            #否则有多少经验用多少经验
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

		batch_memory = self.memory[sample_index, :]

		# q_eval4next is the output of the q_eval network when input s_(t+1)
		# q_next is the output of the q_target network when input s_(s+1)
		# we use q_eval4next to get which action was choosed by eval network in s_(t+1)
		# then we get the Q_value corresponding to that action output by target network  #通过DQN和target网络获取价值，输出是每一个行为的价值
		q_next, q_eval4next = self.q_target(torch.Tensor(batch_memory[:,-self.n_features:])), self.q_eval(torch.Tensor(batch_memory[:,-self.n_features:]))#下一个状态
		q_eval = self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))  #当前状态输入，获取动作价值行为
		#32*1890

		# used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
		#用于计算y，我们需要复制以进行q\u评估，因为此操作可以保持未选择的q\u值不变，
		# so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss
		# 因此，当我们执行qtarget-qeval时，这些Qvalue变为零，不会影响损失的计算
		q_target = torch.Tensor(q_eval.data.numpy().copy())  #复制一份q_eval
		#32*1890
		batch_index = np.arange(self.batch_size, dtype=np.int32)  #0-31是batch_index的索引值
		eval_act_index = batch_memory[:, self.n_features].astype(int)
		#动作的索引值
		reward = torch.Tensor(batch_memory[:, self.n_features+5])   #当前时刻的奖励
		#当前状态的真实奖励
		if self.double_q:
			max_act4next = torch.max(q_eval4next, dim=1)[1]         #为了计算TDtarget
			selected_q_next = q_next[batch_index, max_act4next]
		else:
			selected_q_next = torch.max(q_next, dim=1)[0]

		q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
		#只对评估网络操作
		#损失函数的输入维度  32*1890
		loss = self.loss_func(q_eval, q_target)
		#q_eval是基于当前状态s(n)进行预测
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.cost_his.append(loss)
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1  #学习进度