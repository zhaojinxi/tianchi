import tensorflow

class ConvLSTMCell(tensorflow.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tensorflow.tanh, normalize=True, peephole=True, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tensorflow.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tensorflow.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tensorflow.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    x = tensorflow.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value
    m = 4 * self._filters if self._filters > 1 else 4
    W = tensorflow.get_variable('kernel', self._kernel + [n, m])
    y = tensorflow.nn.convolution(x, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += tensorflow.get_variable('bias', [m], initializer=tensorflow.zeros_initializer())
    j, i, f, o = tensorflow.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tensorflow.get_variable('W_ci', c.shape[1:]) * c
      f += tensorflow.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tensorflow.contrib.layers.layer_norm(j)
      i = tensorflow.contrib.layers.layer_norm(i)
      f = tensorflow.contrib.layers.layer_norm(f)

    f = tensorflow.sigmoid(f + self._forget_bias)
    i = tensorflow.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tensorflow.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tensorflow.contrib.layers.layer_norm(o)
      c = tensorflow.contrib.layers.layer_norm(c)

    o = tensorflow.sigmoid(o)
    h = o * self._activation(c)

    state = tensorflow.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state


class ConvGRUCell(tensorflow.nn.rnn_cell.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, shape, filters, kernel, activation=tensorflow.tanh, normalize=True, data_format='channels_last', reuse=None):
    super(ConvGRUCell, self).__init__(_reuse=reuse)
    self._filters = filters
    self._kernel = kernel
    self._activation = activation
    self._normalize = normalize
    if data_format == 'channels_last':
        self._size = tensorflow.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tensorflow.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def call(self, x, h):
    channels = x.shape[self._feature_axis].value

    with tensorflow.variable_scope('gates'):
      inputs = tensorflow.concat([x, h], axis=self._feature_axis)
      n = channels + self._filters
      m = 2 * self._filters if self._filters > 1 else 2
      W = tensorflow.get_variable('kernel', self._kernel + [n, m])
      y = tensorflow.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
      if self._normalize:
        r, u = tensorflow.split(y, 2, axis=self._feature_axis)
        r = tensorflow.contrib.layers.layer_norm(r)
        u = tensorflow.contrib.layers.layer_norm(u)
      else:
        y += tensorflow.get_variable('bias', [m], initializer=tensorflow.ones_initializer())
        r, u = tensorflow.split(y, 2, axis=self._feature_axis)
      r, u = tensorflow.sigmoid(r), tensorflow.sigmoid(u)

    with tensorflow.variable_scope('candidate'):
      inputs = tensorflow.concat([x, r * h], axis=self._feature_axis)
      n = channels + self._filters
      m = self._filters
      W = tensorflow.get_variable('kernel', self._kernel + [n, m])
      y = tensorflow.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
      if self._normalize:
        y = tensorflow.contrib.layers.layer_norm(y)
      else:
        y += tensorflow.get_variable('bias', [m], initializer=tensorflow.zeros_initializer())
      h = u * h + (1 - u) * self._activation(y)

    return h, h