import numpy as np
import builder
import tensorflow as tf
import HP
import nltk
from ELMo.data import Batcher, TokenBatcher


class Encoder(tf.keras.models.Model):
    def __init__(self, use_character_input=True, max_batch_size=128, max_token_length=50, units=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_char = use_character_input
        self.max_token_length = max_token_length
        self.units = units
        self.ELMo, self.weight_layer = builder.builder(HP.option_file, HP.weight_file,
                                                       use_character_inputs=use_character_input,
                                                       embedding_weight_file=HP.embedding_weight_file,
                                                       max_batch_size=max_batch_size)
        self.rnn = tf.keras.layers.SimpleRNN(units=units)

    def call(self, inputs, sentence_specifier, sentence_num, training=None, mask=None):
        """

        :param inputs: shape: [batch size, par token len] if use_character_inputs is False else [batch size,
        par char len, max token length].
        :param training:
        :param mask:
        :return: [batch_size, max_sent_num, embedding_dim]
        """
        embedding_op = self.ELMo(inputs)
        encoded = self.weight_layer(embedding_op['lm_embeddings'], embedding_op['mask'])
        end = tf.reduce_max(sentence_specifier)
        i = tf.constant(1)
        where_t = tf.where(tf.equal(i, sentence_specifier))
        sent_t = tf.expand_dims(tf.gather_nd(encoded, where_t), axis=0)  # 3D
        max_len = tf.shape(sent_t)[0]
        rnn_mask = tf.constant(True, dtype=tf.bool, shape=[1, max_len])

        def cond(w_mask, sentences, max_len, w_i, w_end):
            return tf.less(w_i, w_end)

        def body(w_mask, sentences, w_max_len, w_i, w_end):
            where = tf.where(tf.equal(w_i, sentence_specifier))
            sent = tf.gather_nd(encoded, where)
            mx = tf.maximum(w_max_len, tf.shape(sent)[0])
            sentences = tf.concat(
                [sentences, tf.constant(-1, shape=[tf.shape(sentences)[0], mx - w_max_len, tf.shape(sentences)[2]])],
                axis=1)
            w_mask = tf.constant(
                [w_mask, tf.constant(False, dtype=tf.bool, shape=[tf.shape(sentences)[0], mx - w_max_len])])
            new_s = tf.expand_dims(
                tf.concat([sent, tf.constant(-1, shape=[mx - tf.shape(sent)[0], tf.shape(sent)[1]])], axis=0), axis=0)
            new_m = tf.concat([tf.constant(True, shape=[1, tf.shape(sent)[0]]),
                               tf.constant(False, shape=[1, mx - tf.shape(sent)[0]])], axis=1)
            sentences = tf.concat([sentences, new_s], axis=0)
            w_mask = tf.concat([w_mask, new_m], axis=0)
            w_max_len = mx
            w_i = w_i + 1
            return w_mask, sentences, w_max_len, w_i, w_end

        rnn_mask, rnn_sentences, _, _, _ = tf.while_loop(cond, body, [rnn_mask, sent_t, max_len, i, end],
                                                         shape_invariants=[tf.TensorShape([None, None]), tf.TensorShape(
                                                             [None, None, sent_t.get_shape()[2]]), max_len.get_shape(),
                                                                           i.get_shape(), end.get_shape()])
        embeddings = self.rnn(inputs=rnn_sentences, mask=rnn_mask)
        max_sent_num = tf.reduce_max(sentence_num)

        j = tf.constant(0)
        ret_mask = tf.constant(False, shape=[0, max_sent_num], dtype=tf.bool)
        ret = tf.constant(-1, shape=[0, max_sent_num, self.units])
        last = tf.constant(0)

        def cond_ret(w_last, w_ret_mask, w_ret, w_j, w_end):
            return tf.less(w_j, w_end)

        def body_ret(w_last, w_ret_mask, w_ret, w_j, w_end):
            r = tf.range(w_last, w_last + tf.gather(sentence_num, w_j))
            rows = tf.concat([tf.gather(embeddings, r),
                              tf.constant(-1, shape=[max_sent_num - tf.gather(sentence_num, w_j), self.units])], axis=0)
            w_ret = tf.concat([w_ret, tf.expand_dims(rows, axis=0)], axis=0)
            w_ret_mask = tf.concat([w_ret_mask, tf.concat([tf.constant(True, shape=[1, tf.gather(sentence_num, w_j)]),
                                                           tf.constant(False, shape=[1,
                                                                                     max_sent_num - tf.gather(
                                                                                         sentence_num,
                                                                                         w_j)])],
                                                          axis=1)], axis=0)
            w_last = w_last + tf.gather(sentence_num, w_j)
            w_j = w_j + 1
            return w_last, w_ret_mask, w_ret, w_j, w_end

        last, ret_mask, ret, j, end = tf.while_loop(cond_ret, body_ret, [last, ret_mask, ret, j, end],
                                                    shape_invariants=[last.get_shape(),
                                                                      tf.TensorShape([None, max_sent_num]),
                                                                      tf.TensorShape([None, max_sent_num, self.units]),
                                                                      j.get_shape(), end.get_shape()])
        return ret, ret_mask