# RESUED CODE FROM https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops


class batch_norm(object):
	"""Code modification of http://stackoverflow.com/a/33950177"""
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		#with tf.variable_scope(name):		
		with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
			self.epsilon = epsilon
			self.momentum = momentum

			self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
			self.name = name

	def __call__(self, x, train=True):
		shape = x.get_shape().as_list()

		if train:
			#with tf.variable_scope(name) as scope:			
			with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as scope:
				self.beta = tf.get_variable("beta", [shape[-1]],
									initializer=tf.constant_initializer(0.))
				self.gamma = tf.get_variable("gamma", [shape[-1]],
									initializer=tf.random_normal_initializer(1., 0.02))
				
				try:
					batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
				except:
					batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')
					
				ema_apply_op = self.ema.apply([batch_mean, batch_var])
				self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

				with tf.control_dependencies([ema_apply_op]):
					mean, var = tf.identity(batch_mean), tf.identity(batch_var)
		else:
			mean, var = self.ema_mean, self.ema_var

		normed = tf.nn.batch_norm_with_global_normalization(
				x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

		return normed

		
def _l2normalize(v, eps=1e-12):
	"""l2 normize the input vector."""
	return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def binary_cross_entropy(preds, targets, name=None):
	"""Computes binary cross entropy given `preds`.
	For brevity, let `x = `, `z = targets`.  The logistic loss is
		loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))
	Args:
		preds: A `Tensor` of type `float32` or `float64`.
		targets: A `Tensor` of the same type and shape as `preds`.
	"""
	eps = 1e-12
	with ops.op_scope([preds, targets], name, "bce_loss") as name:
		preds = ops.convert_to_tensor(preds, name="preds")
		targets = ops.convert_to_tensor(targets, name="targets")
		return tf.reduce_mean(-(targets * tf.log(preds + eps) +
							  (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
	"""Concatenate conditioning vector on feature map axis."""
	x_shapes = x.get_shape()
	y_shapes = y.get_shape()
	return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, 
		   k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
		   name="conv2d"):
	#with tf.variable_scope(name):
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv

def deconv2d(input_, output_shape,
			 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
			 name="deconv2d", with_w=False):
	#with tf.variable_scope(name):
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
							initializer=tf.random_normal_initializer(stddev=stddev))
		
		try:
			deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])

		# Support for verisons of TensorFlow before 0.7.0
		except AttributeError:
			deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if with_w:
			return deconv, w, biases
		else:
			return deconv

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()
	#with tf.variable_scope(scope or "Linear"): original
	with tf.variable_scope(scope or "Linear",reuse=tf.AUTO_REUSE):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias
			
def spectral_normed_weight(weights, num_iters=1, update_collection=None,
                           with_sigma=False):
	"""Performs Spectral Normalization on a weight tensor.
	Specifically it divides the weight tensor by its largest singular value. This
	is intended to stabilize GAN training, by making the discriminator satisfy a
	local 1-Lipschitz constraint.
	Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
	[sn-gan] https://openreview.net/pdf?id=B1QRgziT-
	Args:
    	weights: The weight tensor which requires spectral normalization
    	num_iters: Number of SN iterations.
    	update_collection: The update collection for assigning persisted variable u.
                       If None, the function will update u during the forward
                       pass. Else if the update_collection equals 'NO_OPS', the
                       function will not update the u during the forward. This
                       is useful for the discriminator, since it does not update
                       u in the second pass.
                       Else, it will put the assignment in a collection
                       defined by the user. Then the user need to run the
                       assignment explicitly.
    	with_sigma: For debugging purpose. If True, the fuction returns
                the estimated singular value for the weight tensor.
	Returns:
		w_bar: The normalized weight tensor
		sigma: The estimated singular value for the weight tensor.
	"""
	w_shape = weights.shape.as_list()
	w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
	u = tf.get_variable('u', [1, w_shape[-1]],
                      initializer=tf.truncated_normal_initializer(),
                      trainable=False)
	u_ = u
	for _ in range(num_iters):
		v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
		u_ = _l2normalize(tf.matmul(v_, w_mat))

	sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
	w_mat /= sigma
	if update_collection is None:
		with tf.control_dependencies([u.assign(u_)]):
			w_bar = tf.reshape(w_mat, w_shape)
	else:
		w_bar = tf.reshape(w_mat, w_shape)
		if update_collection != 'NO_OPS':
			tf.add_to_collection(update_collection, u.assign(u_))
	if with_sigma:
		return w_bar, sigma
	else:
		return w_bar


def snconv2d(input_, output_dim,k_h=3, k_w=3, d_h=2, d_w=2,
			sn_iters=1, update_collection=None, name='snconv2d'):
	"""Creates a spectral normalized (SN) convolutional layer.
	Args:
	    input_: 4D input tensor (batch size, height, width, channel).
	    output_dim: Number of features in the output layer.
	    k_h: The height of the convolutional kernel.
	    k_w: The width of the convolutional kernel.
	    d_h: The height stride of the convolutional kernel.
	    d_w: The width stride of the convolutional kernel.
	    sn_iters: The number of SN iterations.
	    update_collection: The update collection used in spectral_normed_weight.
	    name: The name of the variable scope.
	Returns:
		conv: The normalized tensor.
	"""
	with tf.variable_scope(name):
		w = tf.get_variable(
			'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
			initializer=tf.contrib.layers.xavier_initializer())
		w_bar = spectral_normed_weight(w, num_iters=sn_iters,update_collection=update_collection)
		conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
		biases = tf.get_variable('biases', [output_dim],initializer=tf.zeros_initializer())
		conv = tf.nn.bias_add(conv, biases)
		return conv

#new sndeconv2d function added
def sndeconv2d(input_, output_shape,
			 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
			 name="deconv2d", with_w=False,update_collection=None,sn_iters=1):
	#with tf.variable_scope(name):
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
							initializer=tf.contrib.layers.xavier_initializer())
		w_bar = spectral_normed_weight(w, num_iters=sn_iters,update_collection=update_collection)
		try:
			deconv = tf.nn.conv2d_transpose(input_, w_bar, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])

		# Support for verisons of TensorFlow before 0.7.0
		except AttributeError:
			deconv = tf.nn.sndeconv2d(input_, w, output_shape=output_shape,
								strides=[1, d_h, d_w, 1])

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if with_w:
			return deconv, w, biases
		else:
			return deconv
