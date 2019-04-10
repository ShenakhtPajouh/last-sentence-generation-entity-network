import numpy as np
import builder
import tensorflow as tf
import HP
import nltk
from ELMo.data import Batcher, TokenBatcher
from ELMo.keras_model import dump_token_embeddings


class Encoder(tf.keras.models.Model):
    def __init__(self, use_character_input=True, max_batch_size=128, max_token_length=50, units=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_char = use_character_input
        self.max_token_length = max_token_length
        self.units = units
        if use_character_input:
            EWF = None
        else:
            EWF = HP.token_embedding_file
        self.ELMo, self.weight_layer = builder.builder(HP.option_file, HP.weight_file,
                                                       max_token_length=max_token_length,
                                                       use_character_inputs=use_character_input,
                                                       embedding_weight_file=EWF,
                                                       max_batch_size=max_batch_size)
        self.cell = tf.keras.layers.GRUCell(units=units)
        self.rnn = tf.keras.layers.RNN(cell=self.cell, return_sequences=True)

    def call(self, inputs, sentence_specifier, end_sentence_specifier, indices, max_sent_num=20, training=None,
             mask=None):
        """

        :param inputs: of shape [
        :param sentence_specifier:
        :param end_sentence_specifier:
        :param indices:
        :param max_sent_num:
        :param training:
        :param mask:
        :return:
        """
        embedding_op = self.ELMo(inputs)
        encoded = self.weight_layer(embedding_op['lm_embeddings'], embedding_op['mask'])
        # i = tf.constant(1, dtype=tf.int64)
        pars = tf.gather_nd(encoded, sentence_specifier)
        embeddings = self.rnn(inputs=pars)

        sents = tf.gather_nd(embeddings, end_sentence_specifier)

        ret = tf.scatter_nd(shape=[tf.shape(inputs)[0], max_sent_num, self.units], updates=sents, indices=indices)

        return ret


def input_provider(pars, batcher, max_sent_num, use_char_input=True):
    """

    :param pars: array of paragraphs.
    :param batcher: instance of Batcher of TokenBatcher.
    :param max_sent_num: maximum number of sentences.
    :param use_char_input: use char input or token id input.
    :return:
        ret: the input of Encoder. if use_char_input it's from shape [batch size, max paragraph character len, max token
         length of Batcher]; otherwise from shape [batch size, max paragraph token len]. it concatenate all sentence of
         a paragraph.
        ss: sentence specifier for passing to Encoder as sentence_specifier parameter. it's from shape
         [number of all sentences in batch, max len of sentences, 2]. for each sentence it keeps positions of its tokens
         in ret.
        rnn_mask: mask that passing to RNN of Encoder. it is from shape [number of all sentences, max len of sentences]
        indices: it contains positions of sentences in output of Encoder. for example if it's i's element is [j, k],
         then the i's sentence of batch is the k's sentence of j's paragraph.
        sent_mask: mask of sentences of pars. it's from shape [batch size, max sent num]
        end_of_sentences: contains end id of each sentences. it should pass to Encoder as end_sentence_specifier. shape:
         [batch size, 2] ===> for example-> end_of_sentences = [[0 5] [1, 2] [2, 6]] means that first sentence's last
         token is its fifth, for second sentences it's second and for last sentence is sixth token.
    """
    sentence_num = []
    npas = []
    max_sent_len = 0
    max_par_len = 0
    sentence_specifiers = []
    rnn_mask = []
    indices = []
    sent_mask = []
    end_of_sentences = []
    total_sent_counter = 0
    for i in range(len(pars)):
        par = pars[i]
        sent = nltk.sent_tokenize(par)
        sentence_num.append(len(sent))
        batched = [batcher.batch_sentences([nltk.word_tokenize(s)]) for s in sent]
        start_of_sentence = 0
        sent_counter = 0
        for s in batched:
            max_sent_len = max(max_sent_len, np.shape(s)[1])
            end_of_sentences.append(np.expand_dims(np.array([total_sent_counter, np.shape(s)[1] - 1]), axis=0))
            indices.append(np.array([i, sent_counter]))
            sentence_specifiers.append(
                np.concatenate(
                    [np.expand_dims(np.repeat(i, np.shape(s)[1]), axis=1),
                     np.expand_dims(np.arange(start_of_sentence, start_of_sentence + np.shape(s)[1]), axis=1)],
                    axis=1))
            rnn_mask.append(np.repeat(True, np.shape(s)[1]))
            start_of_sentence += np.shape(s)[1]
            sent_counter += 1
            total_sent_counter += 1
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
        sent_mask.append(np.expand_dims(
            np.concatenate([np.repeat(True, len(sent)), np.repeat(False, max_sent_num - len(sent))], axis=0),
            axis=0))

    for i in range(len(sentence_specifiers)):
        sentence_specifiers[i] = np.expand_dims(np.concatenate(
            [sentence_specifiers[i],
             np.full(fill_value=0, dtype=np.int64, shape=[max_sent_len - sentence_specifiers[i].shape[0], 2])]),
            axis=0)
        rnn_mask[i] = np.expand_dims(
            np.concatenate([rnn_mask[i], np.repeat(False, max_sent_len - sentence_specifiers[i].shape[0])], axis=0),
            axis=0)

    for i in range(len(npas)):
        if use_char_input:
            proper_padding = np.repeat(axis=1, a=np.expand_dims(
                np.expand_dims(axis=0, a=np.repeat(0, batcher._max_token_length)), axis=0),
                                       repeats=max_par_len - npas[i].shape[1])
        else:
            proper_padding = np.repeat([[0]], axis=1, repeats=max_par_len - npas[i].shape[1])
        npas[i] = np.concatenate([npas[i], proper_padding], axis=1)
    ret = np.concatenate(npas, axis=0)
    ss = np.concatenate(sentence_specifiers, axis=0)
    return ret, ss, rnn_mask, np.array(indices), np.concatenate(sent_mask, axis=0), np.concatenate(end_of_sentences,
                                                                                                   axis=0)


if __name__ == '__main__':
    dump_token_embeddings(
        HP.vocab_file, HP.option_file, HP.weight_file, HP.token_embedding_file
    )
    tf.reset_default_graph()

    batcher = TokenBatcher(HP.vocab_file)
    inputs, specifier, rnn_mask, indices, sent_mask, end_of_sentences = input_provider(
        ['Pretrained biLMs compute representations useful for NLP tasks . it\'s amazing .',
         'They give state of the art performance for many tasks .'], batcher, max_sent_num=3, use_char_input=False)
    print(inputs)
    print(inputs.shape)
    print(specifier)
    print(specifier.shape)
    print(rnn_mask)
    print(indices)
    print(sent_mask)
    print(end_of_sentences)
    encoder = Encoder(use_character_input=False)
    inp = tf.placeholder(shape=[None, None], dtype=tf.int64)
    ss = tf.placeholder(shape=[None, None, 2], dtype=tf.int64)
    # rnn_mask_placeholder = tf.placeholder(shape=[None], dtype=tf.int64)
    indices_placeholder = tf.placeholder(shape=[None, 2], dtype=tf.int64)
    end_c_p = tf.placeholder(shape=[None, 2], dtype=tf.int64)
    encoded = encoder(inp, ss, end_c_p, indices_placeholder, max_sent_num=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_ = sess.run(encoded, feed_dict={inp: inputs, ss: specifier, end_c_p: end_of_sentences,
                                          indices_placeholder: indices})
