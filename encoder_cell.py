import tensorflow as tf


def encoder_cell(encoder, ELMo_states, encoder_state, id):
    ELMo_cells = encoder.ELMo.lm_graph.lstm_cells['forward']
    encoder_cell = encoder.cell
    inputs = encoder.ELMo.lm_graph.input_pre_process(id)
    ELMo_output_states = []
    for i in range(ELMo_cells):
        inputs, state = ELMo_cells(inputs, ELMo_states[i])
        ELMo_output_states.append(state)
    output, encoder_state = encoder_cell(inputs, encoder_state)

    return output, encoder_state


def get_ELMo_initial_state(lm_graph, batch_size):
    ret = []
    for i in range(lm_graph.n_lstm_layers):
        ret.append(state[:batch_size, :] for state in (lm_graph.init_states['forward'][i]))

    return ret


