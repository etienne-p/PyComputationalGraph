import numpy as np

class OpValue:
	"""
	Notes:
		does not define a backward method,
		as it is meant to be at the terminal nodes in the graph
	"""
	def num_inputs(self):
		return 0
	def assign(self, X):
		self.X = X
	def forward(self):
		return self.X

class OpSum:
	def num_inputs(self):
		return 2
	def forward(self, A, B):
		return A + B
	def backward(self, grad, A, B):
		dA = grad
		dB = grad
		return [dA, dB]

class OpMatMul:
	def num_inputs(self):
		return 2
	def forward(self, A, B):
		return np.matmul(A, B)
	def backward(self, grad, A, B):
		dA = np.matmul(grad, B.T)
		dB = np.matmul(A.T, grad)
		return [dA, dB]

class OpDot:
	def num_inputs(self):
		return 2
	def forward(self, A, B):
		return np.dot(A, B)
	def backward(self, grad, A, B):
		dA = np.dot(grad, B.T)
		dB = np.dot(A.T, grad)
		return [dA, dB]

class OpSigmoid:
	"""
	logistic activation function
	
	Notes:
		even if the op only has one input, we still return gradients in a tuple,
		simplifies backprop code
	"""
	def num_inputs(self):
		return 1
	def forward(self, A):
		return 1. / (1. + np.exp(-A))
	def backward(self, grad, A):
		sig = self.forward(A)
		dA = sig * (1 - sig) * grad
		return [dA]