Answer of Question 1.

os.path.join은 입력받은 문자열들로 1개의 경로를 만드는 코드이다.

Answer of Question 2.

s

Answer of Question 3.

s

Answer of Question 4.

(1) Maxout

import torch
import torch.nn as nn

class DenseBlock(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(DenseBlock, self).__init__()
		self.dense = nn.Linear(in_dim, out_dim)
		self.act = nn.Maxout()

	def forward(self, x):
		out = self.act(self.dense(x))
		return out

(2) ELU

import torch
import torch.nn as nn

class DenseBlock(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(DenseBlock, self).__init__()
		self.dense = nn.Linear(in_dim, out_dim)
		self.act = nn.ELU()

	def forward(self, x):
		out = self.act(self.dense(x))
		return out

(3) GELU

import torch
import torch.nn as nn

class DenseBlock(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(DenseBlock, self).__init__()
		self.dense = nn.Linear(in_dim, out_dim)
		self.act = nn.GELU()

	def forward(self, x):
		out = self.act(self.dense(x))
		return out
