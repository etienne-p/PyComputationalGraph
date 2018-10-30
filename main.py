import numpy as np
from graph import Graph
from prop import prop
from grad_check import gradient_check_graph
from operations import *

g = Graph('my_graph')

X = g.add(OpValue(), 'X') # input
W_1 = g.add(OpValue(), 'W_1') # weight 1
b_1 = g.add(OpValue(), 'b_1') # bias 1
W_2 = g.add(OpValue(), 'W_2') # weight 2
b_2 = g.add(OpValue(), 'b_2') # bias 2
WX_1 = g.add(OpMatMul(), 'WX_1', [X, W_1])
WX_b_1 = g.add(OpSum(), 'WX_b_1', [WX_1, b_1])
sigmoid_1 = g.add(OpSigmoid(), 'sigmoid_1', [WX_b_1])
WX_2 = g.add(OpMatMul(), 'WX_2', [sigmoid_1, W_2])
WX_b_2 = g.add(OpSum(), 'WX_b_2', [WX_2, b_2])
sigmoid_2 = g.add(OpSigmoid(), 'sigmoid_2', [WX_b_2])

g.done()
g.viz()

np.random.seed(2) # for reproducibility

# initialize inputs
parameter_table = {
	'X': np.random.rand(1, 12),
	'W_1': np.random.rand(12, 12),
	'b_1': np.random.rand(12),
	'W_2': np.random.rand(12, 1),
	'b_2': np.random.rand(1)}

# propagation
output, output_table = g.forward(parameter_table)
print('output:', output)
grad_table = g.backward()

print('gradient check, difference: {}'.format(\
	gradient_check_graph(g, parameter_table, grad_table), perturbation=1e-4))


