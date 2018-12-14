# -*- coding: utf-8 -*-
from __future__ import division
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.nccl as nccl

from utils import CSVTracker, StatefulTracker, Timer
from layers import cyclic_learning_rate
from dataset import get_csv_dataset
from model import build_tower

logger = logging.getLogger(__name__)


def train(config, restore=False):
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.gpu_options.allow_growth = True

    with tf.Graph().as_default(), \
         tf.Session(config=sess_config) as sess:
        logger.info("Attempt to load embedding.")
        embedding_init = np.load(config.embed_path).astype(np.float32)
        logger.info("Done.")
        logger.info("Prepare datasets...")

        with open(config.vocab_path, 'r') as fin:
            vocabulary = [line.strip() for line in fin.readlines()]
        vocab_table = tf.contrib.lookup.index_table_from_tensor(vocabulary, default_value=137)  # default is unk
        doc_table = tf.contrib.lookup.index_table_from_tensor(['1', '0', '-1', '-2'], default_value=-1)
        train_set = get_csv_dataset([config.train_path], vocab_table, doc_table, config.batch_size,
                                    num_sub_batch=config.num_gpus, shuffle=True, bucket_width=100)
        train_eval_set = get_csv_dataset([config.train_eval_path], vocab_table, doc_table, config.eval_batch_size,
                                         config.num_gpus, shuffle=False)
        valid_eval_set = get_csv_dataset([config.valid_path], vocab_table, doc_table, config.eval_batch_size,
                                         config.num_gpus, shuffle=False)

        iterator = tf.data.Iterator.from_structure(train_set.output_types, train_set.output_shapes)
        train_iter_init = iterator.make_initializer(train_set)
        train_eval_iter_init = iterator.make_initializer(train_eval_set)
        valid_eval_iter_init = iterator.make_initializer(valid_eval_set)
        logger.info("Done.")

        # build model
        logger.info("Build train graph...")
        tower_grads_list = []
        tower_tvars_list = []
        tower_gvars_list = []
        tower_loss_list = []
        tower_labels_list = []
        tower_oh_preds_list = []

        tower_batches = iterator.get_next()
        for index, tower_batch in enumerate(tower_batches):
            with tf.variable_scope("tower_%d" % index) as scope, \
                    tf.device('/gpu:%d' % index):

                tower_ids, tower_raw_seqs, tower_seqs, tower_lengths, tower_labels = tower_batch
                tower_train_loss, tower_eval_oh_preds, tower_elmo_saver = \
                    build_tower(config, tower_seqs, tower_lengths, tower_labels,
                                initializers={"embedding_init": embedding_init})
                tower_gvars = tf.global_variables(scope._name)
                tower_tvars = tf.trainable_variables(scope._name)
                tower_grads = tf.gradients(tower_train_loss, tower_tvars)
                tower_loss_list.append(tower_train_loss)
                tower_tvars_list.append(tower_tvars)
                tower_gvars_list.append(tower_gvars)
                tower_grads_list.append(tower_grads)

                tower_labels_list.append(tower_labels)
                tower_oh_preds_list.append(tower_eval_oh_preds)
                if index == 0:
                    saver = tf.train.Saver(tower_gvars)
                    elmo_saver = tower_elmo_saver

        with tf.name_scope("tower_gvar_sync"):
            if len(tower_gvars_list) == 1:
                tower_gvar_sync = tf.no_op()
            else:
                sync_ops = []
                for vars in zip(*tower_gvars_list):
                    for var in vars[1:]:
                        sync_ops.append(tf.assign(var, vars[0]))
                tower_gvar_sync = tf.group(*sync_ops)

        with tf.name_scope('all_reduce'):
            avg_tower_grads_list = []
            for grads_to_avg in zip(*tower_grads_list):
                if None in grads_to_avg:
                    avg_tower_grads_list.append(grads_to_avg)
                    continue
                avg_tower_grads_list.append(nccl.all_sum(grads_to_avg))
            avg_tower_grads_list = zip(*avg_tower_grads_list)

        with tf.device('/gpu:0'), tf.name_scope('metrics'):
            # metrics
            labels = tf.concat(tower_labels_list, axis=0)
            # [batch_size, num_aspects, num_labels]
            oh_preds = tf.concat(tower_oh_preds_list, axis=0)
            # [batch_size, num_aspects, num_labels]
            oh_labels = tf.one_hot(labels, depth=4, on_value=True, off_value=False, dtype=tf.bool)
            tps = tf.get_local_variable("tps", shape=[20, 4], dtype=tf.float64)
            fps = tf.get_local_variable("fps", shape=[20, 4], dtype=tf.float64)
            fns = tf.get_local_variable("fns", shape=[20, 4], dtype=tf.float64)

            def cross_and_sum(pred_bool, label_bool):
                cross = tf.logical_and(tf.equal(oh_preds, pred_bool), tf.equal(oh_labels, label_bool))
                return tf.reduce_sum(tf.cast(cross, tf.float64), axis=0)

            f1_updates = tf.group(
                    tf.assign_add(tps, cross_and_sum(pred_bool=True, label_bool=True)),
                    tf.assign_add(fps, cross_and_sum(pred_bool=True, label_bool=False)),
                    tf.assign_add(fns, cross_and_sum(pred_bool=False, label_bool=True)),
            )
            precisions = tps / (tps + fps + 1e-50)
            recalls = tps / (tps + fns + 1e-50)
            f1s = 2 * precisions * recalls / (precisions + recalls + 1e-50)
            macro_f1 = tf.reduce_mean(f1s)
            metrics_update = tf.group(f1_updates)

            # train loss
            loss = tf.add_n(tower_loss_list) / len(tower_loss_list)
        tower_train_ops = []
        for index, (tower_vars, tower_grads) in \
                enumerate(zip(tower_tvars_list, avg_tower_grads_list)):
            with tf.variable_scope("tower_%d" % index), \
                 tf.device('/gpu:%d' % index):
                tower_grads = [grad / len(tower_batches) if grad is not None else None for grad in tower_grads]
                if index == 0:
                    global_step = tf.train.get_or_create_global_step()
                    lr = cyclic_learning_rate(global_step=global_step,
                                              min_lr=0.00005,
                                              max_lr=0.002,
                                              step_size=8205)

                tower_optimizer = tf.contrib.opt.NadamOptimizer(lr)
                tower_grads, _ = tf.clip_by_global_norm(tower_grads, config.grad_clip_max_norm)
                tower_train_op = tower_optimizer.apply_gradients(zip(tower_grads, tower_vars),
                                                                 global_step=global_step if index == 0 else None)
                tower_train_ops.append(tower_train_op)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.group(tower_train_ops)
        logger.info("Done.")

        # start training
        logger.info("Init model...")
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer(),
                  tf.tables_initializer()])
        logger.info("Done.")

        if elmo_saver is not None:
            logger.info("Restoring elmo...")
            elmo_saver.restore(sess, config.elmo_path)
            logger.info("Done.")

        if restore:
            logger.info("Restore model from {}".format(config.model_path))
            saver.restore(sess, config.model_path)
            logger.info("Done.")
        logger.info("Synchronize towers...")
        sess.run(tower_gvar_sync)
        logger.info("Done.")

        fetch_dict = {
            'loss': loss,
            'train_op': train_op,
            'step': global_step,
            'lr': lr
        }
        loss_tracker = CSVTracker(fields=['epoch', 'step', 'loss', 'lr'],
                                  fmts=['%d', '%d', "%.4f", '%g'],
                                  start_time=config.start_time,
                                  log_dir=config.output_dir,
                                  filename='loss')
        acc_tracker = StatefulTracker(cmp_field="valid_f1",
                                      fields=["epoch", "train_f1", "valid_f1", "diff_f1"],
                                      log_dir=config.output_dir,
                                      start_time=config.start_time,
                                      filename='acc')

        def _train(iter_init, epoch):
            sess.run([iter_init])
            fetch = {"epoch": epoch}
            step = sess.run(global_step)
            while True:
                try:
                    if step % 50 == 0:
                        fetch.update(sess.run(fetch_dict))
                        loss_tracker.track(fetch)
                    else:
                        sess.run(train_op)
                    step += 1
                except tf.errors.OutOfRangeError:
                    break

        def _evaluate(iter_init):
            timer = Timer()
            sess.run([iter_init,
                      tf.local_variables_initializer()])
            while True:
                try:
                    sess.run(metrics_update)
                except tf.errors.OutOfRangeError:
                    break
            logger.info("Time elapsed: %s" % timer.tock())
            fetch_macro_f1 = \
                sess.run(macro_f1)
            return fetch_macro_f1

        logger.info("Start training.")
        for epoch in range(config.max_epoch):
            _train(train_iter_init, epoch)
            logger.info("Evaluate train set...")
            train_f1 = _evaluate(train_eval_iter_init)
            logger.info("Evaluate valid set...")
            valid_f1 = _evaluate(valid_eval_iter_init)
            acc_tracker.track(dict(epoch=epoch,
                                   train_f1=train_f1,
                                   valid_f1=valid_f1,
                                   diff_f1=train_f1 - valid_f1))
            if acc_tracker.improved:
                logger.info("Save checkpoint to {}".format(repr(config.model_path)))
                saver.save(sess, config.model_path)
                logger.info("Done.")
            if acc_tracker.staled_tracks > config.early_stop_epoch:
                logger.warning("Stop improve for %d epoch, early stop." % acc_tracker.staled_tracks)
                break
        logger.info("Finish training.")
