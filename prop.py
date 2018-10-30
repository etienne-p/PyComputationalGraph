from functools import reduce
import numpy as np

def prop(nodes, parameter_table):
	"""
	performs forward propagation on a graph passed as a list of nodes
	sorted according to their evaluation order

	Args:
		nodes, dictionary of graph nodes, topologically ordered
		inputs, input values passed as a dictionary whose keys are input nodes uids
	Returns:
		the activation corresponding to the output node
		the table storing activations for each node in the graph
	Notes:
		we'll need these activation to perform the backpropagation pass
	"""
	# assign inputs
	for name, tensor in parameter_table.items():
		nodes[name].op.assign(tensor)
	# propagate, cache activations in a dictionary
	output_table = {}
	for name, node in nodes.items():
		# collect parents activations
		args = [output_table[p.name] for p in node.inputs()]
		output_table[node.name] = node.op.forward(*args)
	# graph output is tha activation of the last internal node
	return output_table[list(nodes.keys())[-1]], output_table


def build_grad(node, output_table, parameter_grad_table, output_grad_table):
	"""
	Computes the gradient corresponding to a node and store it
	think of an 'incoing gradient' which is the sum of the gradients
	backpropagating form each consumer nodes

	therefore we first need to individually compute the gradients from 
	each consumer node

	Args:
		node, the node whose gardient is to be computed
		output_table, a table used to store outputs computed during the forward pass
		parameter_grad_table, stores gradients at nodes input, they may be summed to compute gradients at node outputs
		output_grad_table, a table used to store computed gradients

	Returns:
		the corresponding gradient for the argument node
	"""
	if node.name in output_grad_table:
		return output_grad_table[node.name]
	consumer_gradients = [] # list of gradients corresponding to each consumer node
	for consumer_node in node.consumers():
		# index of node in its consumer node inputs
		parameter_index = consumer_node.inputs().index(node)
		if consumer_node.name in parameter_grad_table:
			consumer_gradients.append(parameter_grad_table[consumer_node.name][parameter_index])
		else:
			consumer_parameter_activations = [output_table[n.name] for n in consumer_node.inputs()]
			# compute the gradient at consumer node output
			consumer_output_gradient = build_grad(consumer_node, output_table, parameter_grad_table, output_grad_table)
			# based on gradient at consumer node output, compute gradients at its inputs
			consumer_parameter_gradients = consumer_node.op.backward(\
				consumer_output_gradient, *consumer_parameter_activations)
			# cache gradients at consumer node inputs
			parameter_grad_table[consumer_node.name] = consumer_parameter_gradients

			# TODO this check really is for debugging only, remove it once fixed
			assert(len(consumer_parameter_gradients) > parameter_index), \
				'parameter_index {} out of range {}'.format(parameter_index, len(consumer_parameter_gradients))

			consumer_gradients.append(consumer_parameter_gradients[parameter_index])
	# gradient at node output is the sum of the gradients at its consumer's inputs
	consumer_gradients_sum = reduce((lambda x, y: x + y), consumer_gradients)
	# cache gradient at node output
	output_grad_table[node.name] = consumer_gradients_sum
	return consumer_gradients_sum

def backprop(nodes, output_table):
	"""
	Performs backpropagation on a list of nodes

	Args:
		nodes, the list of nodes whose gradients are to be computed
		output, forward propagation output, the variable to be differentiated
		output_table, a table used to store outputs computed during the forward pass
	Returns:
		a table storing gradients for argument nodes
	"""
	# stores gradients at nodes output
	output_grad_table = {}
	# stores gradients at nodes input, they may be summed to compute gradients at node outputs
	parameter_grad_table = {} 

	# gradient at all sinks is one, we do not want to handle this special case in ```build_grad()```
	sinks = [n for n in nodes if len(n.consumers()) == 0]
	for n in sinks:
		output_grad_table[n.name] = np.ones(output_table[n.name].shape)

	for node in nodes:
		build_grad(node, output_table, parameter_grad_table, output_grad_table)

	return output_grad_table
