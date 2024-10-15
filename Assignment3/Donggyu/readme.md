Answer of Question 1.

os.path.join은 입력받은 문자열들로 1개의 경로를 만드는 코드이다.

Answer of Question 2.

(1) Autograd : 역전파를 통해 parameter를 업데이트하는 방법
    requires_grad = True : autograd에 모든 operation을 저장하는 코
    loss.backward() : loss에 대하여 .backward()를 호출하는 코드
    with torch.no_grad() : 미분값(gradient)을 사용하지 않도록 설정하는 컨텍스트-관리자(context-manager)

(2) Autograd는 모든 연산들의 기록들을 방향성 비순환 그래프(DAG)에 저장한다. 이때 잎(leaf)은 입력 텐서를, 
    뿌리(root)는 결과 텐서를 나타낸다. root에서 leaf로의 연쇄 법칙에 따라 gradient를 자동으로 계산할 수 있
    다.

    	1. 순전파 : 결괏값을 계산하고 gradient function에 저장한다.
    	2. 역전파 : .grad_fn으로부터 gradient를 계산하여 .grad에 저장하고 잎(leaf)으로 전파(propagate)한
                    다.

(3) torch.nn : 다양한 신경망를 생성할 수 있는 패키지이다.

	1. torch.nn.Module : State를 포함할 수 있는 호출 가능한 오브젝터를 생성
 	2. torch.nn.Parameter : Backpropagation동안 업데이트가 필요한 가중치가 있음을 알려주는 코드.
                                required_grad가 설정된 텐서만 업데이트가 실행됨.
	3. torch.nn.functional : Activation function, loss function는 물론, convolution, linear
                                 layer에 대해서 state를 저장하지 않는 layer를 포함하는 모듈

    torch.optim : Autograd에 저장된 미분값을 이용하여 parameter를 업데이트 하는데 필요한 함수를 가지고 있
                  는 패키지
    torch.utils.data.Dataset : 샘플과 정답을 저장하고, len 및 getitem 이 있는 객체의 추상 인터페이스
    torch.utils.data.DataLoader : 모든 종류의 dataset을 기반으로 데이터의 배치들을 출력하는 반복자
                                  (iterator)를 생성
    
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
