import numpy as np

def arrays_nd_array_1d(arrays):
	"""
	Concatenate a list of ND arrays to a single 1D array
	Args:
		arrays: list of ND arrays
	"""
	flat_list = [np.ndarray.flatten(np.asarray(array)) for array in arrays]
	return np.concatenate(flat_list)

def size_from_shape(shape):
	"""
	Get number of elements in an array based on its shape
	Args:
		shape, tuple representing numpy array shape
	Returns:
		number of elements corresponding to the given shape
	"""
	size = 1
	for d in shape:
		size = size * d
	return size

def array_1d_arrays_nd(array, shapes):
	"""
	Split a 1D array to a list of ND arrays
	Args:
		array, the 1D array to split
		shapes, list of shapes in the returned array list
	Returns:
		list of ND arrays
	"""
	index = 0
	arrays_nd = []
	for shape in shapes:
		size = size_from_shape(shape)
		flat_array = array[index:index + size]
		arrays_nd.append(flat_array.reshape(shape))
		index = index + size
	return arrays_nd

def gradient_check(parameters, grad, forward_func, perturbation=1e-4, equality_threshold=1e-7):
	"""
	Args:
		parameter, input value to the node (or node group)
		grad, gradient obtained via backprop for the node (or node group)
		forward_func, forward propagation operation
		perturbation, scalar by which input will be perturbed to compute gradient via finite difference
		equality_threshold, threshold determining wether or not backprop and 
		finite difference gradients are considered equal
	Notes:
		both args represent tiny scalars and could've been called epsilon,
		so we go for more descriptive names
	"""
	assert(parameters.shape == grad.shape), 'input and gradient should have the same shape'
	# fd stands for Finite Difference
	fd_grad = np.zeros(grad.shape)
	# sequentially perturb each parameter to compute the finite difference gradient
	for i in range(fd_grad.shape[0]):
		parameter_min = parameters.copy()
		parameter_max = parameters.copy()
		parameter_min[i] -= perturbation
		parameter_max[i] += perturbation
		fd_grad[i] = (forward_func(parameter_max) - forward_func(parameter_min)) / (2. * perturbation)

	numerator = np.linalg.norm(fd_grad - grad)
	denominator = np.linalg.norm(fd_grad ) + np.linalg.norm( grad)
	difference = numerator / denominator
	# TMP debug
	return difference # < equality_threshold

def make_forward_flattened(graph, parameter_table):
	"""
	Create a function that will be passed to ```gradient_check()``` 
	to perform forward propagation, working with flattened parameter arrays
	Args:
		graph, computational graph we want to perform gradient check on
		parameter_table, table storing inputs
	Return:
		a function that perform forward propagation on the graph
	"""
	parameter_shapes = {n.name: parameter_table[n.name].shape for n in graph.parameter_nodes}

	def forward_flattened(concat_params):
		parameter_table = {}
		arrays = array_1d_arrays_nd(concat_params, parameter_shapes.values())
		for name, array in zip(parameter_shapes.keys(), arrays):
			parameter_table[name] = array
		output, _ = graph.forward(parameter_table)
		# for gradient check to make sense you need forward prop to ultimately return a scalar
		assert(output.size == 1), 'gradient check requires the forward function to return a scalar'
		return output
	return forward_flattened

def gradient_check_graph(graph, parameter_table, grad_table, perturbation=1e-4, equality_threshold=1e-7):
	"""
	Perform gradient check on a graph
	Args:
		graph, computational graph to perform gradient check on
		parameter_table, a table storing inputs
		parameter_table, a table storing gradient, obtained via backprop
		perturbation, scalar by which input will be perturbed to compute gradient via finite difference
		equality_threshold, threshold determining wether or not backprop and 
		finite difference gradients are considered equal
	"""
	return gradient_check(
		# Note: order of arrays in input and gradient must be consistent
		# we use the order of input nodes in the graph as a reference
		# flatten inputs
		arrays_nd_array_1d([parameter_table[n.name] for n in graph.parameter_nodes]), \
		# flatten gradient
		arrays_nd_array_1d([grad_table[n.name] for n in graph.parameter_nodes]), \
		# create a forward propagation function
		make_forward_flattened(graph, parameter_table), \
		perturbation=perturbation, equality_threshold=equality_threshold)
