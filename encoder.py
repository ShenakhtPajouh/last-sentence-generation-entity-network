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

        :param inputs: it is from shape [batch size, par token len] if use_character_inputs is False else [batch size,
        par char len, max token length].
        :param sentence_specifier: sentence specifier of shape [batch size, par len]
        :param sentence_num: number of sentences for each batch. it's of shape [batch size]
        :param max_sent_len: max len of sentences(token or character)
        :param max_sent_num: max number of sentences in a paragraph(batch)
        :param training:
        :param mask:
        :return: of shape [batch size, max number of sentences, embedding dim]
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


def input_provider(pars, batcher, max_par_len=0, use_char_input=True):
    """

    :param pars: a list contain multiple sentences
    :return: sentences, sentence specifier, max_sent_len, max_sent_num
    """
    sentence_num = []
    last = 0
    npas = []
    max_sent_len = 0
    max_sent_num = 0
    sent_counter = 0
    sentence_specifiers = []
    for par in pars:
        ssnp = np.array([])
        sent = nltk.sent_tokenize(par)
        max_sent_len = max(max_sent_len, max(len(s) for s in sent))
        max_sent_num = max(max_sent_num, len(sent))
        sentence_num.append(len(sent))
        batched = [batcher.batch_sentences([nltk.word_tokenize(s)]) for s in sent]
        for s in batched:
            ssnp = np.concatenate([ssnp, np.repeat(sent_counter, s.shape[1])])
            sent_counter += 1
        encoded_par = np.concatenate(batched, axis=1)
        if use_char_input:
            proper_sop = np.expand_dims(np.expand_dims(np.repeat(HP.sop, batcher._max_token_length), axis=0), axis=0)
            proper_eop = np.expand_dims(np.expand_dims(np.repeat(HP.eop, batcher._max_token_length), axis=0), axis=0)
        else:
            proper_sop = np.array([[batcher._lm_vocab._bop]])
            proper_eop = np.array([[batcher._lm_vocab._eop]])
        encoded_par = np.concatenate([proper_sop, encoded_par, proper_eop], axis=1)
        max_par_len = max(max_par_len, encoded_par.shape[1])
        npas.append(encoded_par)
        sentence_specifiers.append(ssnp)

    for i in range(len(npas)):
        if use_char_input:
            proper_padding = np.repeat(axis=1, a=np.expand_dims(
                np.expand_dims(axis=0, a=np.repeat(0, batcher._max_token_length)), axis=0),
                                       repeats=max_par_len - npas[i].shape[1])
        else:
            proper_padding = np.repeat([[0]], axis=1, repeats=max_par_len - npas[i].shape[1])
        npas[i] = np.concatenate([npas[i], proper_padding], axis=1)
        sentence_specifiers[i] = np.expand_dims(np.concatenate(
            [sentence_specifiers[i], np.repeat(-1, max_par_len - 2 - sentence_specifiers[i].shape[0])]), axis=0)
    ret = np.concatenate(npas, axis=0)
    ss = np.concatenate(sentence_specifiers, axis=0)
    return ret, ss, max_sent_len, max_sent_num


if __name__ == '__main__':
    # tf.enable_eager_execution()
    batcher = Batcher(HP.vocab_file, 50)
    q, w, e, r = input_provider(['Hi! How are you? I\'m fine. thank you'], batcher)
    print(q.shape)
    print(w)
    print(e)
    print(r)
    encoder = Encoder()
    # a = np.random.random(size=(2, 5, 20))
    """
    a = tf.placeholder(shape=[2, 5, 20], dtype=tf.int64)
    b = tf.placeholder(shape=[2, 5], dtype=tf.int64)
    c = tf.placeholder(shape=[2], dtype=tf.int64)
    """
    a = tf.placeholder(shape=[None, None, 50], dtype=tf.int64)
    b = tf.placeholder(shape=[None, None], dtype=tf.int64)
    c = tf.placeholder(shape=[None], dtype=tf.int64)
    # b = np.array([[0, 0, 1, -1, -1], [2, 3, 3, 3, 4]])
    # c = np.array([2, 3])
    x, y = encoder(a, b, c)
    print(x)
    print(y)
