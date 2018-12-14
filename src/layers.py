# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
import numpy as np
import math
import re


def project(inputs, kernel, bias=None):
    shape = tf.shape(inputs)
    shape_list = inputs.shape.as_list()

    if shape_list[-1] == None:
        raise ValueError()
    kernel_shape_list = kernel.shape.as_list()
    if len(kernel_shape_list) != 2:
        raise ValueError()
    if kernel_shape_list[0] != shape_list[-1]:
        raise ValueError()
    outputs = tf.matmul(tf.reshape(inputs, [-1, shape_list[-1]]), kernel)
    if bias is not None:
        outputs = tf.nn.bias_add(outputs, bias)
    outputs_shape_list = [shape[i] if dim is None else dim for i, dim in enumerate(shape_list[:-1])] + \
                         [kernel_shape_list[-1]]
    outputs = tf.reshape(outputs, outputs_shape_list)
    return outputs


def apply_mask(inputs, seq_lens, seq_axis):
    shape = tf.shape(inputs)
    mask_shape = [shape[0]] + [1 for _ in range(seq_axis - 1)] + [shape[seq_axis]] + \
                 [1 for _ in range(len(inputs.shape.as_list()) - seq_axis - 1)]

    output_mask = tf.sequence_mask(seq_lens,
                                   maxlen=shape[seq_axis],
                                   dtype=tf.float32)
    output_mask = tf.reshape(output_mask, mask_shape)
    outputs = inputs * output_mask
    return outputs


def cyclic_learning_rate(global_step,
                         min_lr=0.00005,
                         max_lr=0.0005,
                         step_size=2000.,
                         gamma=0.99994,
                         mode='triangular2',
                         name=None):
    if global_step is None:
        raise ValueError("global_step is required for cyclic_min_lr.")
    with tf.name_scope(name, "CyclicLearningRate",
                       [min_lr, global_step]) as name:
        min_lr = tf.convert_to_tensor(min_lr, name="min_lr")
        dtype = min_lr.dtype
        global_step = tf.cast(global_step, dtype)
        step_size = tf.cast(step_size, dtype)

        def cyclic_lr():
            double_step = tf.multiply(2., step_size)
            global_div_double_step = tf.divide(global_step, double_step)
            cycle = tf.floor(tf.add(1., global_div_double_step))
            double_cycle = tf.multiply(2., cycle)
            global_div_step = tf.divide(global_step, step_size)
            tmp = tf.subtract(global_div_step, double_cycle)
            x = tf.abs(tf.add(1., tmp))
            a1 = tf.maximum(0., tf.subtract(1., x))
            a2 = tf.subtract(max_lr, min_lr)
            clr = tf.multiply(a1, a2)
            if mode == 'triangular2':
                clr = tf.divide(clr, tf.cast(tf.pow(2, tf.cast(
                        cycle - 1, tf.int32)), tf.float32))
            if mode == 'exp_range':
                clr = tf.multiply(tf.pow(gamma, global_step), clr)
            return tf.add(clr, min_lr, name=name)

        return cyclic_lr()


def avg_getter_factory(var_scope=None, decay=0.999):
    with tf.name_scope('avg_vars'):
        tvars = tf.trainable_variables(var_scope if var_scope else None)
        avger = tf.train.ExponentialMovingAverage(decay)
        update_op = avger.apply(tvars)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)

    def avg_getter(getter, name, *_, **__):
        var = getter(name, *_, **__)
        if not var.trainable or 'elmo' in var.name:
            avg_var = var
        else:
            avg_var = avger.average(var)
        return avg_var

    return avger, avg_getter


class DenseLayer(object):
    def __init__(self,
                 num_units,
                 use_bias=True,
                 output_keep_prob=None,
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None):
        self.num_units = num_units
        self.use_bias = use_bias
        self.output_keep_prob = output_keep_prob
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name = self.__class__.__name__ if name is None else name
        self._variables = None
        self.kernel = None
        self.bias = None
        self.built = False

    @property
    def variables(self):
        return self._variables

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)

        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `%s` ' % self.__class__.__name__ +
                             'should be defined. Found `None`.')

        feature_size = input_shape[-1].value
        with tf.variable_scope(self.name, reuse=self.built):
            self.kernel = tf.get_variable("kernel", shape=[feature_size, self.num_units],
                                          initializer=self.kernel_initializer)
            self._variables = [self.kernel]
            if self.use_bias:
                self.bias = tf.get_variable("bias", shape=[self.num_units],
                                            initializer=self.bias_initializer)
                self._variables.append(self.bias)
        self.built = True

    def __call__(self, inputs, training=True):
        # inputs: [batch_size, num_heads, feature_size]
        if not self.built:
            self.build(inputs.shape)
        with tf.variable_scope(self.name, reuse=self.built):
            outputs = project(inputs, self.kernel, self.bias)
            if self.activation is not None:
                outputs = self.activation(outputs)
            if training and self.output_keep_prob is not None:
                outputs = tf.nn.dropout(outputs, keep_prob=self.output_keep_prob)
        return outputs


class CudnnRNNLayer(object):
    def __init__(self,
                 num_units,
                 num_layers,
                 direction="bidirectional",
                 kernel_keep_prob=1.0,
                 output_keep_prob=1.0,
                 cell="lstm",
                 num_projs=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None):

        self.num_units = num_units
        self.num_layers = num_layers
        self.num_projs = num_projs
        self.direction = direction
        self.kernel_keep_prob = kernel_keep_prob
        self.output_keep_prob = output_keep_prob
        self.kernel_initializer = kernel_initializer
        self.name = name

        self.kernel = None
        self.proj_kernel = None

        if cell == "gru":
            layer_class = cudnn_rnn.CudnnGRU
        elif cell == "lstm":
            layer_class = cudnn_rnn.CudnnLSTM
        elif cell == 'basic':
            layer_class = cudnn_rnn.CudnnRNNTanh
        else:
            raise ValueError("Not supported cell type: %s" % cell)

        self.layer = layer_class(num_layers=num_layers,
                                 num_units=num_units,
                                 input_mode="linear_input",
                                 direction=direction,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name=name)
        self.avger = None
        self.built = False

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)
        if input_shape.ndims != 3:
            raise ValueError(
                    'The rank of the inputs to `CudnnLSTMLayer` should be 3, got rank: %d.' % input_shape.ndims)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `CudnnLSTMLayer` should be defined. Found `None`.')
        self.layer.build(input_shape)
        self.kernel = self.layer.kernel
        with tf.variable_scope(self.layer._scope):
            if self.num_projs:
                if self.direction == 'bidirectional':
                    self.proj_kernel = tf.get_variable("proj_kernel", shape=[2 * self.num_units, self.num_projs])
                else:
                    self.proj_kernel = tf.get_variable("proj_kernel", shape=[self.num_units, self.num_projs])

        self.built = True

    def set_avger(self, avger):
        self.avger = avger

    def __call__(self, embed_seqs, seq_lens, initial_state=None, training=True, mask=True):
        if not self.built:
            self.build(embed_seqs.shape)

        with tf.variable_scope(self.layer._scope, reuse=self.built):
            if self.avger is not None:
                kernel = self.avger.average(self.kernel)
            else:
                kernel = self.kernel

            if training and self.kernel_keep_prob < 1.0:
                self.layer.kernel = tf.nn.dropout(kernel, keep_prob=self.kernel_keep_prob)
            else:
                self.layer.kernel = kernel

            embed_seqs = tf.transpose(embed_seqs, [1, 0, 2])
        outputs, final_states = self.layer(embed_seqs, initial_state=initial_state, training=training)
        with tf.variable_scope(self.layer._scope, reuse=self.built):
            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs_shape = tf.shape(outputs)
            batch_size = outputs_shape[0]
            maxlen = outputs_shape[1]
            if self.num_projs:
                if self.direction == 'bidirectional':
                    flat_outputs = tf.matmul(tf.reshape(outputs, [-1, 2 * self.num_units]), self.proj_kernel)
                else:
                    flat_outputs = tf.matmul(tf.reshape(outputs, [-1, self.num_units]), self.proj_kernel)
                outputs = tf.reshape(flat_outputs, [batch_size, maxlen, self.num_projs])
            if training:
                outputs = tf.nn.dropout(outputs, keep_prob=self.output_keep_prob)
            if mask:
                output_mask = tf.expand_dims(
                        tf.sequence_mask(seq_lens,
                                         maxlen=tf.maximum(tf.reduce_max(seq_lens), maxlen),
                                         dtype=tf.float32), axis=-1)
                outputs = outputs * output_mask
        return outputs, final_states


class DropEmbeddingLayer(object):
    def __init__(self,
                 num_words,
                 num_units,
                 embed_keep_prob=None,
                 output_keep_prob=None,
                 variational_dropout=False,
                 kernel_initializer=None,
                 name=None,
                 trainable=True):
        self.num_words = num_words
        self.num_units = num_units
        self.embed_keep_prob = embed_keep_prob
        self.output_keep_prob = output_keep_prob
        self.variational_dropout = variational_dropout
        self.kernel_initializer = kernel_initializer
        self.name = self.__class__.__name__ if name is None else name
        self.trainable = trainable
        if isinstance(self.kernel_initializer, np.ndarray):
            if self.kernel_initializer.shape != (num_words, num_units):
                raise ValueError('Shape mismatch: expected ndarray initializer with shape %s, got %s' %
                                 (str((num_words, num_units)), str(self.kernel_initializer.shape)))

        self.kernel = None
        self.built = False
        self._variables = None

    def variables(self):
        return self._variables

    def build(self):
        with tf.variable_scope(self.name, reuse=self.built):
            if isinstance(self.kernel_initializer, np.ndarray):
                self.kernel = tf.get_variable("kernel",
                                              initializer=self.kernel_initializer,
                                              trainable=self.trainable)
            else:
                self.kernel = tf.get_variable("kernel",
                                              shape=[self.num_words, self.num_units],
                                              initializer=self.kernel_initializer,
                                              dtype=tf.float32,
                                              trainable=self.trainable)
            self._variables = [self.kernel]

        self.built = True

    def __call__(self, seqs, training=True):
        if not self.built:
            self.build()
        with tf.variable_scope(self.name):
            embed_seqs = tf.nn.embedding_lookup(self.kernel, seqs)
            if training:
                if self.embed_keep_prob is not None:
                    embed_seqs = self._embedding_dropout(embed_seqs, seqs, self.embed_keep_prob)
                if self.output_keep_prob is not None:
                    if self.variational_dropout:
                        embed_shape = tf.shape(embed_seqs)
                        embed_seqs = tf.nn.dropout(embed_seqs, self.output_keep_prob,
                                                   noise_shape=[embed_shape[0], 1, embed_shape[2]])
                    else:
                        embed_seqs = tf.nn.dropout(embed_seqs, self.output_keep_prob)
        return embed_seqs

    def _embedding_dropout(self, embeddings, ids, keep_prob, seed=None, name=None):
        with tf.name_scope(name, "embedding_dropout", [embeddings]):
            embeddings = tf.convert_to_tensor(embeddings, name="embeddings")
            shape = tf.shape(embeddings)
            if not embeddings.dtype.is_floating:
                raise ValueError("embeddings has to be a floating point tensor since it's going to"
                                 " be scaled. Got a %s tensor instead." % embeddings.dtype)
            if isinstance(keep_prob, float) and not 0.0 < keep_prob <= 1.0:
                raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                 "range (0, 1], got %g" % keep_prob)

            # Early return if nothing needs to be dropped.
            if isinstance(keep_prob, float) and keep_prob == 1:
                return embeddings
            else:
                keep_prob = tf.convert_to_tensor(
                        keep_prob, dtype=embeddings.dtype, name="keep_prob")
                keep_prob = tf.assert_scalar(keep_prob)

                # Do nothing if we know keep_prob == 1
                if tf.contrib.util.constant_value(keep_prob) == 1:
                    return embeddings

            # uniform [keep_prob, 1.0 + keep_prob)
            random_tensor = keep_prob
            uniq_ids, indices = tf.unique(tf.reshape(ids, [-1]))
            mask_indices = tf.stack([tf.tile(tf.expand_dims(tf.range(shape[0]), axis=-1), [1, shape[1]]),
                                     tf.reshape(indices, [shape[0], shape[1]])], axis=-1)
            rand_mask = tf.random_uniform([shape[0], tf.size(uniq_ids)], seed=seed, dtype=embeddings.dtype)
            random_tensor += tf.gather_nd(rand_mask, mask_indices)
            binary_tensor = tf.floor(random_tensor)
            dropped_embeddings = tf.div(embeddings, keep_prob) * tf.expand_dims(binary_tensor, axis=-1)
            return dropped_embeddings


class MultiHeadAttentivePooling(object):
    def __init__(self,
                 atn_units,
                 num_heads,
                 atn_kernel_keep_prob=None,
                 atn_weight_keep_prob=None,
                 atn_kernel_initializer=None,
                 atn_bias_initializer=tf.zeros_initializer(),
                 head_initializer=None,
                 name=None,
                 trainable=True):
        self.atn_units = atn_units
        self.num_heads = num_heads
        self.atn_kernel_keep_prob = atn_kernel_keep_prob
        self.atn_weight_keep_prob = atn_weight_keep_prob
        self.atn_bias_initializer = atn_bias_initializer
        self.atn_kernel_initializer = atn_kernel_initializer
        self.head_initializer = head_initializer
        self.name = self.__class__.__name__ if name is None else name
        self.trainable = trainable

        self.built = False
        self.atn_kernel = None
        self.atn_bias = None
        self.heads = None
        self.input_units = None
        self._variables = None

    @property
    def variables(self):
        return self._variables

    def key_reg(self):
        with tf.variable_scope(self.name, reuse=self.built):
            if self.num_heads > 1:
                head_sim_matrix = tf.matmul(self.heads, self.heads, transpose_b=True) - tf.eye(self.num_heads)
                atn_reg = tf.square(tf.norm(head_sim_matrix))
            else:
                atn_reg = tf.constant(0.0)
        return atn_reg

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)

        if input_shape.ndims != 3:
            raise ValueError(
                    'The rank of the inputs to `%s` should be 3, got rank: %d.' % (self.__class__.__name__,
                                                                                   input_shape.ndims))

        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `%s` '
                             'should be defined. Found `None`.' % self.__class__.__name__)

        self.input_units = input_shape[-1].value

        with tf.variable_scope(self.name, reuse=self.built):
            self.atn_kernel = tf.get_variable("atn_kernel", shape=[self.input_units, self.atn_units],
                                              initializer=self.atn_kernel_initializer)

            self.atn_bias = tf.get_variable("atn_bias", shape=[self.atn_units],
                                            initializer=self.atn_bias_initializer)
            self.heads = tf.get_variable("heads", shape=[self.num_heads, self.atn_units],
                                         initializer=self.head_initializer)

            self._variables = [self.atn_kernel, self.atn_bias, self.heads]
        self.built = True

    def get_key_seqs(self, embed_seqs, seq_lens, mask=True, training=True):
        if not self.built:
            self.build(embed_seqs.shape)
        with tf.variable_scope(self.name, reuse=self.built):
            # [batch_size, num_steps, embed_size]
            if training and self.atn_kernel_keep_prob is not None:
                atn_kernel = tf.nn.dropout(self.atn_kernel, keep_prob=self.atn_kernel_keep_prob)
            else:
                atn_kernel = self.atn_kernel
            atn_key_seqs = tf.tanh(project(embed_seqs, atn_kernel, bias=self.atn_bias))
            if mask:
                atn_key_seqs = apply_mask(atn_key_seqs, seq_lens, seq_axis=1)
            return atn_key_seqs

    def atn_pool(self, atn_key_seqs, embed_seqs, seq_lens, training=True):
        if not self.built:
            self.build(embed_seqs.shape)

        with tf.variable_scope(self.name, reuse=self.built):
            # [batch_size, num_steps, num_heads]
            sim_seqs = project(atn_key_seqs, tf.transpose(self.heads))
            atn_weights = tf.nn.softmax(sim_seqs, axis=1)
            # masked softmax
            atn_weights = apply_mask(atn_weights, seq_lens, seq_axis=1)
            # [batch_size, num_steps, num_heads]
            atn_weights = atn_weights / tf.reduce_sum(atn_weights, axis=1, keep_dims=True)
            # get output
            # [batch_size, num_heads, num_steps]
            atn_weights = tf.transpose(atn_weights, perm=[0, 2, 1])
            if training and self.atn_weight_keep_prob is not None:
                atn_weights = tf.nn.dropout(atn_weights, keep_prob=self.atn_weight_keep_prob)

            # [batch_size, num_heads, embed_size]
            atn_out = tf.matmul(atn_weights, embed_seqs)
        return atn_out

    def batch_atn_pool(self, batch_heads, atn_key_seqs, embed_seqs, seq_lens, training=True):
        if not self.built:
            self.build(embed_seqs.shape)
        with tf.variable_scope(self.name, reuse=self.built):
            # [batch_size, num_heads, atn_units] batch_heads
            # [batch_size, num_steps, atn_units] atn_key_seqs
            # [batch_size, num_steps, num_heads]
            sim_seqs = tf.matmul(atn_key_seqs, tf.transpose(batch_heads, [0, 2, 1]))
            atn_weights = tf.nn.softmax(sim_seqs, axis=1)
            # masked softmax
            atn_weights = apply_mask(atn_weights, seq_lens, seq_axis=1)
            # [batch_size, num_steps, num_heads]
            atn_weights = atn_weights / tf.reduce_sum(atn_weights, axis=1, keep_dims=True)
            # get output
            # [batch_size, num_heads, num_steps]
            atn_weights = tf.transpose(atn_weights, perm=[0, 2, 1])
            # [batch_size, num_heads, embed_size]
            atn_out = tf.matmul(atn_weights, embed_seqs)
        return atn_out

    def __call__(self, embed_seqs, seq_lens, training=True):
        if not self.built:
            self.build(embed_seqs.shape)
        # [batch_size, num_steps, atn_units]
        atn_key_seqs = self.get_key_seqs(embed_seqs, seq_lens, mask=False, training=training)
        # [batch_size, num_heads, embed_size]
        atn_out = self.atn_pool(atn_key_seqs, embed_seqs, seq_lens, training=training)
        return atn_out


class MultiHeadDenseLayer(object):
    def __init__(self,
                 num_units,
                 num_heads,
                 output_keep_prob=None,
                 activation=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None):
        self.num_units = num_units
        self.num_heads = num_heads
        self.output_keep_prob = output_keep_prob
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name = self.__class__.__name__ if name is None else name
        self._variables = None
        self.kernel = None
        self.bias = None
        self.built = False

    @property
    def variables(self):
        return self._variables

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)

        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `%s` ' % self.__class__.__name__ +
                             'should be defined. Found `None`.')

        feature_size = input_shape[-1].value
        with tf.variable_scope(self.name, reuse=self.built):
            self.kernel = tf.get_variable("kernel", shape=[self.num_heads, feature_size, self.num_units],
                                          initializer=self.kernel_initializer)
            self.bias = tf.get_variable("bias", shape=[self.num_heads, self.num_units],
                                        initializer=self.bias_initializer)

        self._variables = [self.kernel, self.bias]

        self.built = True

    def __call__(self, inputs, training=True):
        # inputs: [batch_size, num_heads, feature_size]
        if not self.built:
            self.build(inputs.shape)
        with tf.variable_scope(self.name, reuse=self.built):
            features = tf.transpose(inputs, [1, 0, 2])  # [num_heads, batch_size, feature_size]
            # [num_heads, batch_size, num_units]
            outputs = tf.matmul(features, self.kernel) + tf.expand_dims(self.bias, axis=1)
            if self.activation is not None:
                outputs = self.activation(outputs)
            outputs = tf.transpose(outputs, [1, 0, 2])  # [batch_size, num_heads, num_units]

            if training and self.output_keep_prob is not None:
                outputs = tf.nn.dropout(outputs, keep_prob=self.output_keep_prob)
        return outputs


class MultiHeadAngularProjection(object):
    def __init__(self,
                 num_units,
                 num_heads,
                 margin=0.1,
                 scale=30.,
                 kernel_initializer=None,
                 name=None):
        self.num_units = num_units
        self.num_heads = num_heads
        self.margin = margin
        self.scale = scale
        self.kernel_initializer = kernel_initializer
        self.name = self.__class__.__name__ if name is None else name
        self._variables = None
        self.kernel = None
        self.built = False

    @property
    def variables(self):
        return self._variables

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)

        if input_shape.ndims != 3:
            raise ValueError(
                    'The rank of the inputs to `AMSoftmax` should be 3, got rank: %d.' % input_shape.ndims)

        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `AMSoftmax` '
                             'should be defined. Found `None`.')

        feature_size = input_shape[-1].value
        with tf.variable_scope(self.name, reuse=self.built):
            self.kernel = tf.get_variable("kernel", shape=[self.num_heads, feature_size, self.num_units],
                                          initializer=self.kernel_initializer)
            self.kernel = tf.nn.l2_normalize(self.kernel, dim=1)
        self._variables = [self.kernel]

        self.built = True

    def __call__(self, features):
        # features: [batch_size, num_heads, feature_size]
        if not self.built:
            self.build(features.shape)
        with tf.variable_scope(self.name, reuse=self.built):
            features = tf.nn.l2_normalize(features, dim=-1)
            features = tf.transpose(features, [1, 0, 2])  # [num_heads, batch_size, feature_size]
            logits = tf.matmul(features, self.kernel)  # [num_heads, batch_size, num_units]
            logits = tf.transpose(logits, [1, 0, 2])  # [batch_size, num_heads, num_units]
            logits = tf.clip_by_value(logits, -1, 1)  # numerical stability
        return logits

    def softmax(self, one_hot_labels, logits, type='arcface', label_smooth_epsilon=None):
        # batch_size, num_heads, num_units
        if type == 'arcface':
            cos_m, sin_m = math.cos(self.margin), math.sin(self.margin)
            cos_logits = logits
            sin_logits = tf.sqrt(1 - cos_logits ** 2)
            cos_logits_add_m = cos_m * cos_logits - sin_m * sin_logits
            special_val = cos_logits - math.sin(math.pi - self.margin) * self.margin
            target_logits = tf.where(cos_logits > math.cos(math.pi - self.margin), cos_logits_add_m,
                                     special_val)
            logits = self.scale * tf.where(tf.equal(one_hot_labels, 1), target_logits, logits)

        elif type == 'amsoftmax':
            logits = self.scale * tf.where(tf.equal(one_hot_labels, 1), logits - self.margin, logits)

        else:
            raise ValueError("Not supported angular softmax.")

        if label_smooth_epsilon is not None:
            one_hot_labels = ((1 - label_smooth_epsilon) * one_hot_labels) + (
                    label_smooth_epsilon / self.num_units)

        losses = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)
        return losses


class HighwayLayer(object):
    def __init__(self, num_units, num_layers, output_keep_prob=None, name=None):
        self.gate_denses = [DenseLayer(num_units, use_bias=True, activation=tf.sigmoid, name='gate_%d' % i) for i in
                            range(num_layers)]
        self.hidden_denses = [DenseLayer(num_units, use_bias=True, activation=tf.tanh, name="hidden_%d" % i) for i in
                              range(num_layers)]

        self.name = self.__class__.__name__ if name is None else name
        self.num_units = num_units
        self.output_keep_prob = output_keep_prob
        self.built = False

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)

        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `AMSoftmax` '
                             'should be defined. Found `None`.')

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for gate_dense in self.gate_denses:
                gate_dense.build([None, self.num_units])
            for hidden_dense in self.hidden_denses:
                hidden_dense.build([None, self.num_units])

    def __call__(self, inputs, training=False):
        if not self.built:
            self.build(inputs.shape)
        new_inputs = inputs
        for gate_dense, hidden_dense in zip(self.gate_denses, self.hidden_denses):
            gate = gate_dense(new_inputs)
            hidden = hidden_dense(new_inputs)
            new_inputs = gate * new_inputs + (1 - gate) * hidden
            if training and self.output_keep_prob is not None:
                new_inputs = tf.nn.dropout(new_inputs, self.output_keep_prob)
        return new_inputs


class ELMoLayer(object):
    def __init__(self, vocab_size, embed_size, hidden_size, cell_type, num_layers, l2_weight):

        self.embedding_layer = DropEmbeddingLayer(vocab_size,
                                                  embed_size)
        self.fw_layers = [CudnnRNNLayer(num_units=hidden_size,
                                        num_layers=1,
                                        direction="unidirectional",
                                        num_projs=embed_size,
                                        cell=cell_type,
                                        name='fw_rnn_%d' % i) for i in range(num_layers)]
        self.bw_layers = [CudnnRNNLayer(num_units=hidden_size,
                                        num_layers=1,
                                        direction="unidirectional",
                                        num_projs=embed_size,
                                        cell=cell_type,
                                        name='bw_rnn_%d' % i) for i in range(num_layers)]

        self.l2_weight = l2_weight
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.elmo_w = None
        self.elmo_gamma = None
        self._elmo_saver = None
        self.built = False

    def reg(self):
        return self.l2_weight * tf.reduce_sum(tf.square(self.elmo_w))

    @property
    def saver(self):
        return self._elmo_saver

    def build(self):
        with tf.variable_scope('elmo', reuse=tf.AUTO_REUSE):
            self.embedding_layer.build()
            for layer in self.fw_layers:
                layer.build([None, None, self.embed_size])
            for layer in self.bw_layers:
                layer.build([None, None, self.embed_size])

            num_layers = len(self.fw_layers)

            def clean(name):
                name = re.sub('^tower_\d+/', '', name)
                name = re.sub(':\d+$', '', name)
                return name

            elmo_vars = tf.trainable_variables(tf.get_variable_scope()._name)
            self._elmo_saver = tf.train.Saver({clean(var.name): var for var in elmo_vars})
            self.elmo_w = tf.get_variable('elmo_w',
                                          shape=(num_layers,),
                                          initializer=tf.zeros_initializer(),
                                          trainable=True)
            self.elmo_gamma = tf.get_variable(
                    'elmo_gamma',
                    shape=(1,),
                    initializer=tf.ones_initializer,
                    regularizer=None,
                    trainable=True,
            )
            self.built = True

    def __call__(self, seqs, lengths):
        if not self.built:
            self.build()

        with tf.variable_scope('elmo', reuse=tf.AUTO_REUSE):
            embed_seqs = self.embedding_layer(seqs, training=False)
            # fw
            fw_outputs_list = [embed_seqs]
            fw_inputs = embed_seqs
            for fw_layer in self.fw_layers:
                fw_outputs, _ = fw_layer(fw_inputs, lengths, training=False, mask=True)
                fw_inputs = fw_outputs
                fw_outputs_list.append(fw_outputs)

            # bw
            bw_outputs_list = [embed_seqs]
            bw_inputs = tf.reverse_sequence(embed_seqs, lengths, seq_axis=1)
            for bw_layer in self.bw_layers:
                bw_outputs, _ = bw_layer(bw_inputs, lengths, training=False, mask=True)
                bw_inputs = bw_outputs
                bw_outputs_list.append(tf.reverse_sequence(bw_outputs, lengths, seq_axis=1))

            normed_weights = tf.split(
                    tf.nn.softmax(self.elmo_w + 1.0 / self.num_layers), self.num_layers)

            elmo_feat_seq_list = [weight * tf.stop_gradient(tf.concat([fw_outputs, bw_outputs], axis=-1))
                                  for fw_outputs, bw_outputs, weight in
                                  zip(fw_outputs_list, bw_outputs_list, normed_weights)]
            elmo_feat_seq = tf.add_n(elmo_feat_seq_list)
            output_mask = tf.expand_dims(
                    tf.sequence_mask(lengths,
                                     maxlen=tf.maximum(tf.reduce_max(lengths), tf.shape(embed_seqs)[1]),
                                     dtype=tf.float32), axis=-1)
            elmo_feat_seq = elmo_feat_seq * output_mask * self.elmo_gamma
        return elmo_feat_seq


class MacLayer(object):
    def __init__(self, num_units, num_heads, num_steps, name=None):
        self.num_units = num_units
        self.num_heads = num_heads
        self.num_steps = num_steps
        self.name = name if name is not None else self.__class__.__name__

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.denses = {
                "encode": tf.layers.Dense(self.num_units, name="encode"),
                # control denses
                "control_update": tf.layers.Dense(self.num_units, name="control_update"),
                # read denses
                "read_seq_key": tf.layers.Dense(self.num_units, name="read_seq_key"),
                "read_mem_key": tf.layers.Dense(self.num_units, name="read_mem_key"),
                "read_comp_key": tf.layers.Dense(self.num_units, name="read_comp_key"),
                "read_atn": tf.layers.Dense(1, name="read_atn"),
                # write denses
                "write_new_mem": tf.layers.Dense(self.num_units, name="write_new_mem"),
                "write_hist_atn": tf.layers.Dense(1, name="write_hist_atn"),
                "write_mem_map": tf.layers.Dense(self.num_units, name="write_mem_map"),
                "write_hist_mem_map": tf.layers.Dense(self.num_units, name="write_hist_mem_map"),
                "write_gate": tf.layers.Dense(1, name="write_gate"),
            }

            self.ctrl_inits = tf.get_variable("ctrl_inits", shape=[self.num_heads, self.num_units])
            self.mem_inits = tf.get_variable("mem_inits", shape=[self.num_heads, self.num_units])

    def __call__(self, embed_seqs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            batch_size = tf.shape(embed_seqs)[0]
            embed_seqs = self.denses['encode'](embed_seqs)
            ctrls = tf.tile(tf.expand_dims(self.ctrl_inits, 0), [batch_size, 1, 1])
            mems = tf.tile(tf.expand_dims(self.mem_inits, 0), [batch_size, 1, 1])
            ctrl_history = tf.expand_dims(ctrls, axis=0)
            mem_history = tf.expand_dims(mems, axis=0)
            for i in range(self.num_steps):
                with tf.name_scope("step_%d" % i):
                    new_ctrls = self.control(ctrls)
                    retrieves = self.read(embed_seqs, new_ctrls, mems)
                    new_mems = self.write(ctrls, mems, retrieves, ctrl_history, mem_history)

                    ctrl_history = tf.concat([ctrl_history, tf.expand_dims(new_ctrls, axis=0)], axis=0)
                    mem_history = tf.concat([mem_history, tf.expand_dims(new_mems, axis=0)], axis=0)
                    ctrls = new_ctrls
                    mems = new_mems
        return mems

    def control(self, ctrls):
        # [batch_size, num_heads, num_units]
        new_ctrls = self.denses['control_update'](ctrls)
        return new_ctrls

    def read(self, embed_seqs, ctrls, mems):
        # embed_seqs [batch_size, seq_len, num_units]
        # ctrls [batch_size, num_heads, num_units]
        # mems [batch_size, num_heads, num_units]

        # map embed_seq to keys
        seq_keys = self.denses['read_seq_key'](embed_seqs)  # [batch_size, seq_len, num_units]
        tile_seq_keys = tf.tile(tf.expand_dims(seq_keys, axis=0),
                                [self.num_heads, 1, 1, 1])  # [num_heads, batch_size, seq_len, num_units]
        tile_seq_keys = tf.transpose(tile_seq_keys, [1, 0, 2, 3])  # [batch_size, num_heads, seq_len, num_units]

        # map mems to keys
        mem_key = self.denses['read_mem_key'](mems)  # [batch_size, num_heads, num_units]
        expand_mem_key = tf.expand_dims(mem_key, axis=2)  # [batch_size, num_heads, 1, num_units]

        # compress (mem_key ~ seq_key, seq_key)
        mem_seq_interact = tile_seq_keys * expand_mem_key
        concat_keys = tf.concat([mem_seq_interact,
                                 tile_seq_keys], axis=-1)  # [batch_size, num_heads, seq_len, 2*num_units]

        comp_keys = self.denses['read_comp_key'](concat_keys)  # [batch_size, num_heads, seq_len, num_units]

        # use ctrl to atn
        expand_ctrls = tf.expand_dims(ctrls, axis=2)  # [batch_size, num_heads, 1, num_units]
        ctrl_key_interact = expand_ctrls * comp_keys  # [batch_size, num_heads, seq_len, num_units]
        atn = tf.nn.softmax(self.denses['read_atn'](ctrl_key_interact),
                            axis=1)  # [batch_size, num_heads, seq_len, 1]

        # weighted mean
        expand_embed_seqs = tf.expand_dims(embed_seqs, axis=1)  # [batch_size, 1, seq_len, num_units]
        retrieves = tf.reduce_sum(atn * expand_embed_seqs, axis=2)  # [batch_size, num_heads, num_units]
        return retrieves

    def write(self, ctrls, mems, retrieves, ctrl_history, mem_history):
        # ctrls [batch_size, num_heads, num_units]
        # mems [batch_size, num_heads, num_units]
        # retrieves [batch_size, num_heads, num_units]

        # [batch_size, num_heads, num_units]
        new_mems = self.denses['write_new_mem'](tf.concat([mems, retrieves], axis=-1))

        # optional self attention
        # ctrl_history [reason_step, batch_size, num_heads, num_units]
        ctrl_history_interacts = tf.expand_dims(ctrls, axis=0) * ctrl_history

        # ctrl_atn_dense units=1
        hist_atn = self.denses['write_hist_atn'](
                ctrl_history_interacts)  # [reason_step, batch_size, num_heads, 1]
        hist_atn = tf.nn.softmax(hist_atn, axis=0)  # [reason_step, batch_size, num_heads, 1]
        # mem_history [reason_step, batch_size, num_heads, num_units]
        hist_mems = tf.reduce_sum(hist_atn * mem_history, axis=0)  # [batch_size, num_heads, num_units]

        new_mems = self.denses['write_mem_map'](new_mems) + self.denses['write_hist_mem_map'](hist_mems)

        # optional gating
        gate = tf.sigmoid(self.denses['write_gate'](ctrls))
        new_mems = gate * mems + (1 - gate) * new_mems
        return new_mems


class InterHeadAttention(object):
    def __init__(self,
                 num_units,
                 key_keep_prob=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None):
        self.num_units = num_units
        self.key_keep_prob = key_keep_prob
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.name = self.__class__.__name__ if name is None else name

        self.key_dense = tf.layers.Dense(num_units, activation=tf.tanh,
                                         kernel_initializer=self.kernel_initializer,
                                         bias_initializer=self.bias_initializer)
        self._variables = None

        self.built = False

    @property
    def variables(self):
        return self._variables

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)

        if input_shape.ndims != 3:
            raise ValueError(
                    'The rank of the inputs to `%s` should be 3, got rank: %d.' % (self.__class__.__name__,
                                                                                   input_shape.ndims))

        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `%s` ' % self.__class__.__name__ +
                             'should be defined. Found `None`.')

        with tf.variable_scope(self.name, reuse=self.built):
            self.key_dense.build(input_shape)

        self._variables = self.key_dense.variables
        self.built = True

    def __call__(self, inputs, training=True):
        # inputs: [batch_size, num_heads, feature_size]
        if not self.built:
            self.build(inputs.shape)
        with tf.variable_scope(self.name, reuse=self.built):
            num_heads = tf.shape(inputs)[1]
            input_keys = self.key_dense(inputs)  # [batch_size, num_heads, num_units]
            if training and self.key_keep_prob is not None:
                input_keys = tf.nn.dropout(input_keys, keep_prob=self.key_keep_prob)
            sim_matrix = tf.matmul(input_keys, tf.transpose(input_keys, perm=[0, 2, 1]))
            sim_matrix = tf.nn.softmax(sim_matrix * (1.0 - tf.eye(num_heads)), axis=-1)
            outputs = tf.matmul(sim_matrix, input_keys)
        return outputs


class MultiHeadSelfAttention(object):
    def __init__(self,
                 key_units,
                 num_heads,
                 head_units,
                 key_kernel_initializer=None,
                 key_bias_initializer=None,
                 head_matrix_initializer=None,
                 name=None,
                 trainable=True):

        self.key_kernel_initializer = key_kernel_initializer
        self.key_bias_initializer = key_bias_initializer
        self.head_matrix_initializer = head_matrix_initializer
        self.key_units = key_units
        self.head_units = head_units
        self.num_heads = num_heads
        self.name = self.__class__.__name__ if name is None else name
        self.trainable = trainable

        self.input_units = None
        self.built = False
        self.key_kernel = None
        self.key_bias = None

        self.head_matrix = None
        self._variables = None

    @property
    def variables(self):
        return self._variables

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)

        if input_shape.ndims != 3:
            raise ValueError(
                    'The rank of the inputs to `%s` should be 3, got rank: %d.' % (self.__class__.__name__,
                                                                                   input_shape.ndims))

        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `%s` '
                             'should be defined. Found `None`.' % self.__class__.__name__)

        self.input_units = input_shape[-1].value

        with tf.variable_scope(self.name, reuse=self.built):
            self.key_kernel = tf.get_variable("key_kernel",
                                              shape=[self.head_units + self.input_units, self.key_units],
                                              initializer=self.key_kernel_initializer)
            self.key_bias = tf.get_variable("key_bias", shape=[self.key_units],
                                            initializer=self.key_bias_initializer)

            self.head_matrix = tf.get_variable("head_matrix", shape=[self.num_heads, self.head_units],
                                               initializer=self.head_matrix_initializer)
            self._variables = [self.head_matrix, self.key_kernel, self.key_bias]
        self.built = True

    def __call__(self, embed_seqs, seq_lens, training=True, split=True):
        if not self.built:
            self.build(embed_seqs.shape)

        with tf.variable_scope(self.name, reuse=self.built):
            input_shape = tf.shape(embed_seqs)
            batch_size = input_shape[0]
            maxlen = input_shape[1]
            # [batch_size, num_heads, num_steps, head_units]
            tiled_heads = tf.tile(tf.reshape(self.head_matrix, [1, self.num_heads, 1, self.head_units]),
                                  [batch_size, 1, maxlen, 1])
            # [batch_size, num_heads, num_steps, embed_size]
            tiled_embed_seqs = tf.tile(tf.expand_dims(embed_seqs, axis=1), [1, self.num_heads, 1, 1])
            # [batch_size, num_heads, num_steps, embed_size + head_units]
            key_features = tf.concat([tiled_heads, tiled_embed_seqs], axis=-1)
            # [batch_size, num_heads, num_steps, key_units]
            keys = tf.tanh(tf.nn.bias_add(tf.tensordot(key_features, self.key_kernel, [[3], [0]]),
                                          self.key_bias))
            # [batch_size, num_heads, num_steps, num_steps]
            sims = tf.matmul(keys, tf.transpose(keys, [0, 1, 3, 2]))

            # masked softmax
            # [batch_size, 1, 1, num_steps]
            mask = tf.expand_dims(tf.expand_dims(
                    tf.sequence_mask(seq_lens, maxlen=maxlen, dtype=tf.float32), axis=1), axis=1)
            # [batch_size, 1, num_steps, num_steps]
            mask = mask * tf.transpose(mask, perm=[0, 1, 3, 2])
            # [1, 1, num_steps, num_steps]
            eye_mask = (1 - tf.expand_dims(tf.expand_dims(tf.eye(maxlen), axis=0), axis=0))
            # [batch_size, 1, num_steps, num_steps]
            mask = mask * eye_mask

            # for numerical stability
            # [batch_size, num_heads, num_steps, num_steps]
            atn_weights = tf.nn.softmax(sims, axis=-1)
            atn_weights = atn_weights * mask
            # [batch_size, num_heads, num_steps, num_steps]
            atn_weights = atn_weights / (tf.reduce_sum(atn_weights, axis=-1, keep_dims=True) + 1e-36)
            # [batch_size, num_heads, num_steps, embed_size]
            self_atn_seqs = tf.matmul(atn_weights, tiled_embed_seqs)
            if split:
                atn_seqs_list = [tf.squeeze(seqs, axis=1) for seqs in tf.split(self_atn_seqs,
                                                                               num_or_size_splits=self.num_heads,
                                                                               axis=1)]
                return atn_seqs_list
        return self_atn_seqs


class SelfAttention(object):
    def __init__(self,
                 key_units,
                 head_units,
                 key_kernel_initializer=None,
                 key_bias_initializer=None,
                 head_key_initializer=None,
                 name=None,
                 trainable=True):

        self.key_kernel_initializer = key_kernel_initializer
        self.key_bias_initializer = key_bias_initializer
        self.head_key_initializer = head_key_initializer
        self.key_units = key_units
        self.head_units = head_units
        self.name = self.__class__.__name__ if name is None else name
        self.trainable = trainable

        self.input_units = None
        self.built = False
        self.key_kernel = None
        self.key_bias = None

        self.head_key = None
        self._variables = None

    @property
    def variables(self):
        return self._variables

    def build(self, input_shape):
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)

        if input_shape.ndims != 3:
            raise ValueError(
                    'The rank of the inputs to `%s` should be 3, got rank: %d.' % (self.__class__.__name__,
                                                                                   input_shape.ndims))

        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `%s` '
                             'should be defined. Found `None`.' % self.__class__.__name__)

        self.input_units = input_shape[-1].value

        with tf.variable_scope(self.name, reuse=self.built):
            self.key_kernel = tf.get_variable("key_kernel",
                                              shape=[self.head_units + self.input_units, self.key_units],
                                              initializer=self.key_kernel_initializer)
            self.key_bias = tf.get_variable("key_bias", shape=[self.key_units],
                                            initializer=self.key_bias_initializer)

            self.head_key = tf.get_variable("head_key", shape=[self.head_units],
                                            initializer=self.head_key_initializer)
            self._variables = [self.head_key, self.key_kernel, self.key_bias]
        self.built = True

    def __call__(self, embed_seqs, seq_lens, training=True):
        if not self.built:
            self.build(embed_seqs.shape)

        with tf.variable_scope(self.name, reuse=self.built):
            input_shape = tf.shape(embed_seqs)
            batch_size = input_shape[0]
            maxlen = input_shape[1]
            # [batch_size, num_steps, head_units]
            tiled_heads = tf.tile(tf.reshape(self.head_key, [1, 1, self.head_units]),
                                  [batch_size, maxlen, 1])
            # [batch_size, num_steps, embed_size + head_units]
            key_features = tf.concat([tiled_heads, embed_seqs], axis=-1)
            # [batch_size, num_steps, key_units]
            keys = tf.tanh(
                    tf.nn.bias_add(tf.tensordot(key_features, self.key_kernel, [[2], [0]]), self.key_bias))
            # [batch_size, num_steps, num_steps]
            sims = tf.matmul(keys, tf.transpose(keys, [0, 2, 1]))

            # masked softmax
            # [batch_size, num_steps, 1]
            mask = tf.expand_dims(tf.sequence_mask(seq_lens, maxlen=maxlen, dtype=tf.float32), axis=-1)
            # [batch_size, num_steps, num_steps]
            mask = mask * tf.transpose(mask, perm=[0, 2, 1])
            # [1, num_steps, num_steps]
            eye_mask = tf.expand_dims(1 - tf.eye(maxlen), axis=0)
            # [batch_size, num_steps, num_steps]
            mask = mask * eye_mask
            # for numerical stability
            atn_weights = tf.nn.softmax(sims, axis=-1)
            atn_weights = atn_weights * mask
            # [batch_size, num_steps, num_steps]
            atn_weights = atn_weights / (tf.reduce_sum(atn_weights, axis=-1, keep_dims=True) + 1e-36)
            # [batch_size, num_steps, embed_size]
            self_atn_seqs = tf.matmul(atn_weights, embed_seqs)

        return self_atn_seqs
