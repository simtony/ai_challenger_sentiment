from __future__ import division

import logging
import tensorflow as tf
from layers import DropEmbeddingLayer, CudnnRNNLayer, MultiHeadAttentivePooling, DenseLayer, \
    avg_getter_factory, ELMoLayer, HighwayLayer

logger = logging.getLogger(__name__)


def get_smooth_label(one_hot_label, epsilon=0.1):
    depth = one_hot_label.get_shape().as_list()[-1]
    return ((1 - epsilon) * one_hot_label) + (epsilon / depth)


def get_f1_loss(y_true, y_pred, epsilon=1e-8):
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - tf.reduce_mean(tf.multiply((1 - f1), f1))


def build_tower(config, seqs, lengths, labels, initializers={}):
    embedding = DropEmbeddingLayer(config.vocab_size,
                                   config.embed_size,
                                   output_keep_prob=config.keep_prob,
                                   kernel_initializer=initializers.get("embedding_init"),
                                   trainable=False)
    if config.use_elmo:
        elmo = ELMoLayer(vocab_size=config.vocab_size,
                         embed_size=300,
                         hidden_size=1024,
                         cell_type='lstm',
                         num_layers=2,
                         l2_weight=0.1)

    rnn = CudnnRNNLayer(num_units=config.hidden_size,
                        num_layers=config.num_layers,
                        direction="bidirectional",
                        kernel_keep_prob=config.rnn_kernel_keep_prob,
                        output_keep_prob=config.keep_prob,
                        cell='lstm',
                        name='rnn')

    poolers = [MultiHeadAttentivePooling(atn_units=config.atn_units,
                                         num_heads=1,
                                         atn_kernel_keep_prob=config.keep_prob,
                                         atn_weight_keep_prob=config.keep_prob,
                                         name="pooler_%d" % i) for i in range(config.num_aspects)]

    highway = HighwayLayer(num_units=config.hidden_size * 2,
                           num_layers=2,
                           output_keep_prob=config.keep_prob,
                           name='highway')

    dense = DenseLayer(num_units=4, name="dense")

    def aspect_logits(seqs, lengths, training=False):
        embed_seqs = embedding(seqs, training=training)
        if config.use_elmo:
            elmo_seqs = elmo(seqs, lengths)
            embed_seqs = tf.concat([elmo_seqs, embed_seqs], axis=-1)
        rnn_feat_seqs, _ = rnn(embed_seqs, lengths, training=training)
        feat_list = []
        for pooler in poolers:
            feat = pooler(rnn_feat_seqs, lengths, training=training)
            feat = tf.nn.dropout(feat, keep_prob=config.keep_prob)
            feat_list.append(feat)
        feats = tf.concat(feat_list, axis=1)
        feats = highway(feats, training=training)
        logits = dense(feats)
        return logits

    smooth_labels = get_smooth_label(tf.one_hot(labels, depth=4, dtype=tf.float32))
    train_logits = aspect_logits(seqs, lengths, training=True)
    train_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=smooth_labels))
    train_loss += get_f1_loss(smooth_labels, tf.nn.softmax(train_logits))

    if config.use_elmo:
        train_loss += elmo.reg()

    vs = tf.get_variable_scope()
    avger, avg_getter = avg_getter_factory()
    vs.set_custom_getter(avg_getter)
    vs.reuse_variables()
    embedding.build()
    rnn.set_avger(avger)
    for pooler in poolers:
        pooler.build([None, None, config.hidden_size * 2])
    dense.build([None, config.hidden_size * 2])

    eval_logits = aspect_logits(seqs, lengths, training=False)
    eval_oh_preds = tf.one_hot(tf.argmax(eval_logits, axis=-1),
                               depth=4,
                               on_value=True, off_value=False, dtype=tf.bool)
    if config.use_elmo:
        return train_loss, eval_oh_preds, elmo.saver
    else:
        return train_loss, eval_oh_preds, None
