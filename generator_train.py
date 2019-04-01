import tensorflow as tf
import HP
import numpy as np
from builder import builder
import RecEntModel as Model
from ELMo.data import Batcher
import nltk


class Generator(tf.keras.models.Model):
    def __init__(self, entity_num, entity_embedding_dim, rnn_hidden_size, vocab_size, max_sent_num=1,
                 max_token_len=50, embedding_matrix=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batcher = Batcher(HP.vocab_file, max_token_len)
        self.bilm_model, self.elmo = builder(options_file=HP.option_file, weight_file=HP.weight_file,
                                             session=tf.Session())
        """
        self.encoder = Model.BasicRecurrentEntityEncoder(embedding_matrix=embedding_matrix, max_entity_num=entity_num,
                                                         entity_embedding_dim=entity_embedding_dim)
        self.decoder = Model.RNNRecurrentEntitiyDecoder(embedding_matrix=embedding_matrix, build_entity_cell=False,
                                                        rnn_hidden_size=rnn_hidden_size,
                                                        vocab_size=vocab_size, max_sent_num=max_sent_num,
                                                        entity_embedding_dim=entity_embedding_dim,
                                                        max_entity_num=entity_num)
        """
        temp = np.zeros([max_token_len], dtype=np.int32)
        temp[:] = 261
        temp[0] = 259
        temp[2] = 260
        self.sop = temp
        self.eop = temp
        self.sop[1] = 262
        self.eop[1] = 263
        self.sop = np.expand_dims(np.expand_dims(self.sop, axis=0), axis=1)
        self.eop = np.expand_dims(np.expand_dims(self.eop, axis=0), axis=1)
        self.max_token_len = max_token_len

    def call(self, inputs, training=None, mask=None):
        """
        inputs: contain batch_p and batch_s. batch_p is array of length batch_size and each cell contain sentences of a paragraph.
        """
        batch_p, batch_s = inputs
        lengthes = []
        pb = []  # batch of paragraphs
        ps = []  # batch of sentences
        for i in range(len(batch_p)):
            lengthes.append([])
            p = batch_p[i]
            s = batch_s[i]
            tokenized_p = [nltk.tokenize.word_tokenize(sent) for sent in p]
            tokenized_sent = nltk.tokenize.word_tokenize(s)
            embed_p = [self.batcher.batch_sentences([sent]) for sent in tokenized_p]
            for pars in embed_p:
                lengthes[i].append(pars.shape[1])
            sentence = self.batcher.batch_sentences([tokenized_sent])
            print()
            par = np.concatenate(embed_p, axis=1)
            print(par.shape)
            par = np.concatenate([self.sop + np.zeros(shape=par.shape), par, self.eop + np.zeros(shape=par.shape)],
                                 axis=1)
            print(par.shape)
            print(sentence.shape)
            print()
            pb.append(par)
            ps.append(sentence)

        max_length_p = max(par.shape[1] for par in pb)
        max_length_s = max(s.shape[1] for s in ps)

        paragraph_batch_ph = tf.placeholder(dtype=np.int32, shape=[len(batch_p), max_length_p, self.max_token_len])
        sentence_batch_ph = tf.placeholder(dtype=np.int32, shape=[len(batch_s), max_length_s, self.max_token_len])

        paragraph_batch = np.zeros(dtype=np.int32, shape=[len(batch_p), max_length_p, self.max_token_len])
        sentence_batch = np.zeros(dtype=np.int32, shape=[len(batch_s), max_length_s, self.max_token_len])

        for i in range(len(batch_p)):
            print(pb[i].shape)
            print(sentence_batch[i, :ps[i].shape[1], :].shape)
            paragraph_batch[i, :pb[i].shape[1], :] = pb[i]
            sentence_batch[i, :ps[i].shape[1], :] = ps[i]

        p_bilm_output = self.bilm_model(paragraph_batch_ph)
        s_bilm_output = self.bilm_model(sentence_batch_ph)

        p_elmo_output = self.elmo(p_bilm_output['lm_embeddings'], p_bilm_output['mask'])
        s_elmo_output = self.elmo(s_bilm_output['lm_embeddings'], s_bilm_output['mask'])

        print(p_elmo_output.shape)
        print(s_elmo_output.shape)
        return p_elmo_output, s_elmo_output


def get_sentence_batch(file, batch_size):
    line = file.readline()
    ret_p = []
    ret_s = []
    is_file_ended = True
    while line:
        args = line.split("\t")
        ret_p.append(nltk.sent_tokenize(args[2]))
        ret_s.append(args[3])
        batch_size -= 1
        if batch_size > 0:
            line = file.readline()
        else:
            break
    if line:
        is_file_ended = False
    return ret_p, ret_s, is_file_ended


file = open(HP.prediction_data_set_path + 'prediction_train.tsv', 'r')
file.readline()

print("opened")

p, s, end = get_sentence_batch(file, 32)

generator = Generator(entity_num=10, entity_embedding_dim=20, rnn_hidden_size=15, vocab_size=10)

print(generator([p, s]))
