import numpy as np
from scipy.misc import derivative
from scipy.special import erf, softmax


class Activation:
	"""
	Class for several activation functions.
	Implemented activation functions are.

	- RELU
	- tanh (Hyperbolic Tangent)
	- Sigmoid (Logistic Function)
	- Linear
	- arctan
	- Error Function (erf)
	- Softmax
	"""
	
	@classmethod
	def relu(cls, x):
		return np.maximum(0, x)

	@classmethod
	def tanh(cls, x):
		return np.tanh(x)

	@classmethod
	def sigmoid(cls, x):
		return 1 / (1 + np.exp(-x))

	@classmethod
	def linear(cls, x):
		return x

	@classmethod
	def arctan(cls, x):
		return np.arctan(x)

	@classmethod
	def erf(cls, x):
		return erf(x)

	@classmethod
	def softmax(cls, x):
		return softmax(x)
	

class Loss:
	"""
	Class for several loss functions.
	Implemented loss functions are:

	- MSE (Mean Squared Error)
	- MAE (Mean Absolute Error)
	- Binary Cross Entropy
	"""
	
	@classmethod
	def mse(cls, t_p, o_p):
		m = t_p.shape[0]
		return (1/m) * (t_p - o_p)**2

	@classmethod
	def mae(cls, t_p, o_p):
		m = t_p.shape[0]
		return (1/m) * np.abs(t_p - o_p)


	@classmethod
	def binary_crossentropy(cls, t_p, o_p):
		return -(t_p*np.log(o_p) + (1-t_p)*np.log(1-o_p))


class NeuralNet:
	"""
	Simple feed-forward neural network model.

	Parameters
	----------
	layers : list<int> or list<tuple>
		List of layer sizes or list of tuples of layer sizes and activations.
	activations : str, function, list<str>, list<functions> or list of mixed str or functions, default=None
		Activation functions for each layer. If None, linear activation is applied.

	 Examples
	--------

	>>> my_layers = [10, 5, 1]
	>>> my_activations = ['relu', np.tanh, my_activation_func]
	>>> model = NeuralNet(my_layers, my_activations)
	>>> model.fit(X, y, **kwargs)
	>>>
	>>> # Make predicitons by using `predict` method
	>>> model.predict(X, **kwargs)
	"""

	def __init__(self, layers: list, activations=None):
		
		if activations is not None:
			assert len(layers) == len(activations)
		else:
			activations = ['linear' for _ in range(len(layers))]

		self.L = len(layers)

		check_all_tuple = all([isinstance(l, tuple) and len(l) == 2 for l in layers])
		if check_all_tuple:
			layers = [l[0] for l in layers]
			activations = [l[1] for l in layers]

		_activations = self._check_activations(activations)
		
		self.layers = layers.copy()
		self.activations = _activations

	@property
	def weights(self) -> (list, list):
		"""
		Return a tuple containing list of weights and biases
		"""
		return self.__W, self.__T

	@weights.setter
	def weights(self, val):
		raise ValueError("Weights does not support assignment.")

	def _check_activations(self, activations):
		_activations = activations.copy()
		check_type = [isinstance(a, str) or callable(a) for a in activations]

		if not check_type:
			raise ValueError("'activations' must be a list of callables or strings")

		for i, a in enumerate(_activations):
			if isinstance(a, str):
				try:
					activation = getattr(Activation, a)
				except:
					raise NotImplementedError(f"Activation function {a} is not implemented, try using a udf instead.")
				else:
					del _activations[i]
					_activations.insert(i, activation)

		return _activations

	def error(self, t_p, o_p):
		E_p = self.loss(t_p, o_p)
		return np.sum(E_p)

	def initialize_weights(self, layers: list):
		self.__W = [np.random.normal(0, 0.05, size=(layers[i], s)) for i, s in enumerate(layers[1:])]
		self.__T = [np.zeros((1, s)) for s in layers[1:]]

	@staticmethod
	def forward_pass(o_p, w, t, activation_func):
		net_p = np.matmul(o_p, w) + t
		o_p = activation_func(net_p)
		return o_p, net_p

	def forward_propagate(self, input_data):
		o_ps, net_ps = [input_data], [input_data]

		o_p = input_data

		for layer in range(self.L):
			w, t = self.__W[layer], self.__T[layer]
			o_p, net_p = self.forward_pass(o_p, w, t, activation_func=self.activations[layer])
			o_ps.append(o_p); net_ps.append(net_p)

		return o_ps, net_ps

	@staticmethod
	def backward_pass(f, net_p, delta_p, w):
		dnet_p = derivative(f, net_p)
		delta_w = np.matmul(delta_p, w.T)
		return dnet_p * delta_w

	def backward_propagate(self, o_ps, net_ps, t_p):
		m = t_p.shape[0]
		dW, dT = [], []

		# Initialize the recursion with the output layer
		do_p = derivative(lambda x: self.loss(t_p, x), o_ps[-1])
		dnet_p = derivative(self.activations[-1], net_ps[-1])
		delta_p = do_p * dnet_p

		for layer in reversed(range(self.L)):
			dw = np.matmul(o_ps[layer].T, delta_p)
			dt = np.sum(delta_p, axis=0, keepdims=True)

			if layer != 0: # No need to compute delta_0 (skipped for performance)
				w = self.__W[layer]
				delta_p = self.backward_pass(f=self.activations[layer], net_p=net_ps[layer], delta_p=delta_p, w=w)

			dW.insert(0, dw); dT.insert(0, dt)

		return dW, dT

	def update_weights(self, dW, dT, eta):
		i = 0
		while i < len(self.__W):
			self.__W[i] -= eta * dW[i]
			self.__T[i] -= eta * dT[i]

			i += 1

	def fit(self, X, y, epochs: int = 50, learning_rate: float = 0.1, loss='mse', verbose=1):
		input_size = X.shape[1]
		self.history = {}
		self.history['costs'] = []
		
		if callable(loss):
			self.loss = loss
		else:
			if isinstance(loss, str):
				self.loss = getattr(Loss, loss)
			else:
				raise NotImplementedError(f"Loss function {loss} is not implemented, try using a udf instead.")
	
		self.layers.insert(0, input_size)
		self.activations.insert(0, Activation.linear)
		self.initialize_weights(self.layers)

		y = y.reshape(len(y), 1)

		for epoch in range(epochs):
			o_ps, net_ps = self.forward_propagate(X)
			E = self.error(t_p=y, o_p=o_ps[-1])
			self.history['costs'].append(E)
			dW, dT = self.backward_propagate(o_ps, net_ps, y)
			self.update_weights(dW, dT, learning_rate)

			if verbose >= 1 and epoch % int(epochs / np.log(epochs)) == 0:
				print(f'Epoch {epoch} \t Cost: {E}')


	def predict(self, X, return_classes=False):
		o_ps, _ = self.forward_propagate(X)

		if return_classes:
			return np.vectorize(lambda x: 1 if x>0.5 else 0)(o_ps[-1])

		return o_ps[-1]
