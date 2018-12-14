# -*- coding: utf-8 -*-
import tensorflow as tf


def get_csv_dataset(filenames,
                    vocab_table,
                    doc_table,
                    batch_size,
                    num_sub_batch=None,
                    buffer_size=30000,
                    bucket_width=None,
                    shuffle=False,
                    repeat=None):
    dataset = tf.contrib.data.CsvDataset(filenames,
                                         [tf.int32, tf.string, tf.string] + [tf.constant(['']) for _ in range(20)],
                                         buffer_size=buffer_size,
                                         header=True,
                                         field_delim=',',
                                         use_quote_delim=True,
                                         na_value='')
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    if repeat is not None:
        dataset = dataset.repeat(repeat)

    def parse_record(id, yi_seq, _, *args):
        words = tf.string_split([tf.regex_replace(yi_seq, "\n", ' ')]).values
        word_ids = tf.to_int32(vocab_table.lookup(words))
        labels = tf.stack(args)
        labels = tf.to_int32(doc_table.lookup(labels))
        return id, words, word_ids, labels

    dataset = dataset.map(parse_record)
    dataset = dataset.map(lambda id, words, word_ids, labels: (id, words, word_ids, tf.size(word_ids), labels),
                          num_parallel_calls=4)

    # bucketing
    if bucket_width is not None:
        def key_func(id, words, word_ids, length, labels):
            bucket_id = length // bucket_width
            return tf.to_int64(bucket_id)

        def reduce_func(unused_key, windowed_data):
            return windowed_data.padded_batch(batch_size,
                                              padded_shapes=(
                                                  tf.TensorShape([]),
                                                  tf.TensorShape([None]),
                                                  tf.TensorShape([None]),
                                                  tf.TensorShape([]),
                                                  tf.TensorShape([20])),
                                              padding_values=(0, '', 0, 0, 0))

        dataset = dataset.apply(
                tf.contrib.data.group_by_window(
                        key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
    else:
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=(
                                           tf.TensorShape([]),
                                           tf.TensorShape([None]),
                                           tf.TensorShape([None]),
                                           tf.TensorShape([]),
                                           tf.TensorShape([20])),
                                       padding_values=(0, '', 0, 0, 0))

    if num_sub_batch is not None:
        dataset = dataset.filter(
                lambda id_batch, words_batch, word_ids_batch, length_batch, labels_batch:
                tf.shape(id_batch)[0] >= num_sub_batch)

        def split_batches(num_splits, batches):
            batch_size = tf.shape(batches[0])[0]
            # evenly distributed sizes
            divisible_sizes = tf.fill([num_splits], tf.floor_div(batch_size, num_splits))
            remainder_sizes = tf.sequence_mask(tf.mod(batch_size, num_splits),
                                               maxlen=num_splits,
                                               dtype=tf.int32)
            frag_sizes = divisible_sizes + remainder_sizes

            batch_frags_list = []
            for batch in batches:
                batch_frags = tf.split(batch, frag_sizes, axis=0)
                batch_frags_list.append(batch_frags)

            frag_batches_list = zip(*batch_frags_list)
            # fix corner case
            for i, frag_batches in enumerate(frag_batches_list):
                if len(frag_batches) == 1:
                    frag_batches_list[i] = frag_batches[0]
            return frag_batches_list

        dataset = dataset.map(lambda id_batch, words_batch, word_ids_batch, length_batch, labels_batch:
                              split_batches(num_sub_batch,
                                            [id_batch, words_batch, word_ids_batch, length_batch, labels_batch]),
                              num_parallel_calls=4)
    dataset = dataset.prefetch(10)
    return dataset
