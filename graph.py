from graphviz import Digraph
from prop import prop, backprop

# note: we append an underscore to variables whose most sensible names are already in use
# we optimize for lisibility
# design choice: we try to remain as close as possible to the book's algorithm
# we currently assume that a graph only has one output node, that may change


# design decision: put forward and backward passes in the same class or in 2 separate classes
# also, no storage at the node level, 
# so that we may decide to use them without storage in production easily
# implementing a num_input on each node might be tedious but it has the merit of simplicity

class Node:
	"""
	Represents a node in the computational graph,
	holds its connexions and operation 
	(which is responsible for providing forward and backward passes)

	Note:
		children will be set by the ```update_children()``` procedure
		```inputs()``` and ```consumers()``` are syntactic sugar for algorithm lisibility
	"""
	def __init__(self, name, op, parents=None):
		self.name = name
		self.op = op
		self.parents = [] if parents is None else parents
		self.children = [] 
	def inputs(self):
		return self.parents
	def consumers(self):
		return self.children

def update_children(nodes):
	"""
	set nodes children in a graph where node parents are known

	Args:
		nodes, list of all nodes belonging to a graph
	"""
	for node in nodes:
		if node.parents is not None:
			for p in node.parents:
				p.children.append(node)

# TODO add support for multiple outputs
#
def topological_sort(nodes):
	"""
	Sorts graph nodes according to their evaluation order, 
	as the propagation and backpropagation procedures expect to process
	sequences of nodes

	Args:
		modes, list of nodes belonging to the graph
	Returns:
		graph nodes sorted according to their evaluation order,
		splitted in two lists, parameter_nodes and internal_nodes,
		as we'll need to identify input nodes to assign them input values
	Note:
		this implementation is not optimized for performance, which,
		considering that topological_sort will occur only once in an experiment,
		is not a big deal
	"""
	# start by collecting input nodes,
	# they, by definition, are all independent,
	# therefore the order of nodes in parameter_nodes does not matter
	parameter_nodes = [n for n in nodes if len(n.parents)==0]
	assert(len(parameter_nodes) > 0), 'topological_sort failed, no input nodes found'
	# now collect other nodes, and sort them topologically
	unsorted_internal_nodes = [n for n in nodes if n not in parameter_nodes]
	internal_nodes = []
	# iteratively remove sinks to perform topological sort
	while len(unsorted_internal_nodes) > 0:
		for n in unsorted_internal_nodes:
			# we consider a sink a node whose children, if any, 
			# are not in unsorted_internal_nodes
			is_sink = True
			for c in n.children:
				if c in unsorted_internal_nodes:
					is_sink = False
					break
			if is_sink:
				unsorted_internal_nodes.remove(n)
				internal_nodes.append(n)
	# reverse internal nodes as we want first nodes in the list to be the ones
	# that'll be evaluated first, that is, we want sinks to be at the end of the list
	return parameter_nodes, internal_nodes[::-1]

class Graph:

	def __init__(self, name):
		self.name = name
		self.nodes = {}
		self.parameter_nodes = None
		# forward propagation cache values
		self.output_table = None

	def add(self, op, name, inputs=None):
		# make sure name is unique
		if name in self.nodes:
			raise Exception('node name {} already in use'.format(name))
		# make sure that right number of inputs was provided
		if not (inputs is None and op.num_inputs() == 0 \
			or inputs is not None and op.num_inputs() == len(inputs)):
			raise Exception('Inputs do not match, expected {}, received {}'\
				.format(str(op.num_inputs()), str(0 if inputs is None else len(inputs))))
		
		input_nodes = None
		if inputs is not None:
			# collect parents based on their name if strings were passed
			if len(inputs) > 0 and type(inputs[0]) == str:
				input_nodes = [self.nodes[n] for n in inputs]
			else:
				input_nodes = inputs
		node = Node(name, op, input_nodes)
		self.nodes[name] = node
		return node

	def done(self):
		"""
		to be called once all nodes have been added to the graph,
		that is, when we are ready to set children based on parents,
		and perform topological sort
		"""
		update_children(self.nodes.values())
		self.parameter_nodes, internal_nodes = topological_sort(self.nodes.values())
		# topologically reorder nodes dictionary
		self.nodes = {n.name: n for n in self.parameter_nodes}
		for n in internal_nodes:
			self.nodes[n.name] = n

	def viz(self):
		"""Vizualise using GraphViz"""
		assert(self.parameter_nodes is not None), \
			'viz failed, you may have forgotten to call .done() on the graph'
		g = Digraph('G', filename='viz/'+self.name, format='png')
		for node in self.nodes.values():
			node_str = node.name if self.output_table is None else '{}, {}'\
				.format(node.name, str(self.output_table[node.name].shape))
			g.node(node.name, node_str)
		for node in self.nodes.values():
			for p in node.parents:
				g.edge(p.name, node.name)
		g.view()

	def forward(self, parameter_table):
		# make sure a value has been provided for each input
		for i in self.parameter_nodes:
			if i.name not in parameter_table:
				raise Exception('missing value for parameter {}'.format(i.name))
		# we store output_table as we'll use it for backprop and viz
		output, self.output_table = prop(self.nodes, parameter_table)
		return output, self.output_table

	def backward(self):
		if self.output_table is None:
			raise Exception('no output_table, \
				you may be attempting to backprop without having first performed forward propagation')
		return backprop(list(self.nodes.values()), self.output_table)