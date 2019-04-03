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
        if use_character_input:
            EWF = None
        else:
            EWF = HP.embedding_weight_file
        self.ELMo, self.weight_layer = builder.builder(HP.option_file, HP.weight_file,
                                                       max_token_length=max_token_length,
                                                       use_character_inputs=use_character_input,
                                                       embedding_weight_file=EWF,
                                                       max_batch_size=max_batch_size)
        self.rnn = tf.keras.layers.SimpleRNN(units=units)

    def call(self, inputs, sentence_specifier, sentence_num, max_sent_len=100, max_sent_num=20, training=None,
             mask=None):
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
        i = tf.constant(1, dtype=tf.int64)
        where_t = tf.where(tf.equal(i, sentence_specifier))
        sent_t = tf.gather_nd(encoded, where_t)  # 2D
        sh = tf.shape(sent_t)[0]
        print(tf.expand_dims(tf.range(sh), axis=1))
        print(sent_t)
        print(sent_t.shape[-1])
        sent_t = tf.expand_dims(
            tf.scatter_nd(shape=[max_sent_len, sent_t.shape[-1]], indices=tf.expand_dims(tf.range(sh), axis=1),
                          updates=sent_t), axis=0)
        print("done")
        rnn_mask = tf.transpose(tf.scatter_nd(shape=[max_sent_len, 1], indices=tf.expand_dims(tf.range(sh), axis=1),
                                              updates=tf.constant(True, shape=[1, 1])))

        def cond(w_mask, sentences, w_i, w_end):
            return tf.less(w_i, w_end)

        def body(w_mask, sentences, w_i, w_end):
            where = tf.where(tf.equal(w_i, sentence_specifier))
            sent = tf.gather_nd(encoded, where)
            new_s = tf.scatter_nd(shape=[1, max_sent_len, sent_t.shape[-1]],
                                  indices=tf.expand_dims(tf.range(tf.shape(sent)[1]), axis=1),
                                  updates=tf.expand_dims(sent, axis=0))
            new_m = tf.transpose(
                tf.scatter_nd(shape=[max_sent_len, 1], indices=tf.expand_dims(tf.range(tf.shape(sent)[1]), axis=1),
                              updates=tf.constant(True, shape=[1, 1])))
            sentences = tf.concat([sentences, new_s], axis=0)
            w_mask = tf.concat([w_mask, new_m], axis=0)
            w_i = w_i + 1
            return w_mask, sentences, w_i, w_end

        rnn_mask, rnn_sentences, _, _ = tf.while_loop(cond=cond, body=body, loop_vars=[rnn_mask, sent_t, i, end],
                                                      shape_invariants=[tf.TensorShape([None, max_sent_len]),
                                                                        tf.TensorShape(
                                                                            [None, max_sent_len,
                                                                             sent_t.shape[-1]]),
                                                                        i.get_shape(), end.get_shape()])
        embeddings = self.rnn(inputs=rnn_sentences, mask=rnn_mask)

        j = tf.constant(0, dtype=tf.int64)
        ret_mask = tf.constant(False, shape=[0, max_sent_num], dtype=tf.bool)
        ret = tf.constant(-1, shape=[0, max_sent_num, self.units])
        last = tf.constant(0, dtype=tf.int64)

        def cond_ret(w_last, w_ret_mask, w_ret, w_j, w_end):
            return tf.less(w_j, w_end)

        def body_ret(w_last, w_ret_mask, w_ret, w_j, w_end):
            r = tf.range(w_last, w_last + tf.gather(sentence_num, w_j))
            rows = tf.scatter_nd(shape=[max_sent_num, self.units],
                                 indices=tf.expand_dims(tf.range(tf.shape(r)[0]), axis=0),
                                 updates=tf.gather(embeddings, r))
            w_ret = tf.concat([w_ret, tf.expand_dims(rows, axis=0)], axis=0)
            mask_t = tf.scatter_nd(shape=[max_sent_num], indices=tf.expand_dims(tf.range(tf.shape(r)[0]), axis=0),
                                   updates=tf.constant(True, shape=[1]))
            w_ret_mask = tf.concat([w_ret_mask, tf.expand_dims(mask_t, axis=0)], axis=0)
            w_last = w_last + tf.gather(sentence_num, w_j)
            w_j = w_j + 1
            return w_last, w_ret_mask, w_ret, w_j, w_end

        last, ret_mask, ret, j, end = tf.while_loop(cond_ret, body_ret, [last, ret_mask, ret, j, end],
                                                    shape_invariants=[last.get_shape(),
                                                                      tf.TensorShape([None, max_sent_num]),
                                                                      tf.TensorShape([None, max_sent_num, self.units]),
                                                                      j.get_shape(), end.get_shape()])
        return ret, ret_mask
