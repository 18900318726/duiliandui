#这个应该是对seq2seq模型结构的设置，设置了encoder，decoder，
# 采用了birnn以及attention，还包括了embedding过程

import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.python.layers import core as layers_core

#定义了多层RNN的网络结构，RNN的单元结构为LSTM结构，每层输出进行dropout操作。
def getLayeredCell(layer_size, num_units, input_keep_prob,output_keep_prob=1.0):
    #rnn.MultiRNNCell定义了多层的循环神经网络，其中里面的rnn单元  #rnn.BasicLSTMCell(num_units)这个函数定义了一层lstm单元，num_unit指这一层的隐单元个数。
    return rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(num_units),input_keep_prob, output_keep_prob) for i in range(layer_size)])
                                                                                                        #for i in range(layer_size)]建立了多层
#双向RNN网络，并对网络的输出进行处理。
def bi_encoder(embed_input, in_seq_len, num_units, layer_size, input_keep_prob):
    # bi_layer_size是双向rnn中每一个方向的层数
    bi_layer_size = int(layer_size / 2)
    # 调用自定义的getLayeredCell函数，返回双向RNN需要传入的RNNCell实例
    #正向合反向lstm神经网络
    encode_cell_fw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
    encode_cell_bw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
    # 调用函数tf.nn.bidirectional_dynamic_rnn，创建一个双向循环神经网络。


    bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = encode_cell_fw,  #这两个是默认参数，函数会自动将输入反向后传给cell_bw神经网络计算
            cell_bw = encode_cell_bw,
            inputs = embed_input,  #网络输入，一个长度为T的list，list中的每个Tensor元素shape为[batch_size,input_size]
            sequence_length = in_seq_len,  #一个int32/int64的向量，长度为[batch_size],包含每个序列的实际长度
            dtype = embed_input.dtype,  #初始状态的数据类型
            time_major = False)   # 假设输入的训练数据（128batch,28steps,128hidden）其中time_major是说的时间点28steps是否位于X_in中第一个维度
                                  # 所以此时为false包含前向和后向RNN输出的张量

    # tf.nn.bidirectional_dynamic_rnn的输出是一个一个(outputs,output_state_fw,output_state_bw)的元组
    # 因此将encoder_output做concat，然后将encoder_state赋值转成元组形式返回。
    encoder_output = tf.concat(bi_encoder_output, -1)
    encoder_state = []
    for layer_id in range(bi_layer_size):
        encoder_state.append(bi_encoder_state[0][layer_id])
        encoder_state.append(bi_encoder_state[1][layer_id])
    encoder_state = tuple(encoder_state)
    return encoder_output, encoder_state

#加“注意力”的解码器
def attention_decoder_cell(encoder_output, in_seq_len, num_units, layer_size,input_keep_prob):
    # 选择注意力机制(这里提供了两种不同的机制BahdanauAttention和LuongAttenion，区别在于计算的方式有点区别)；
    attention_mechanim = tf.contrib.seq2seq.BahdanauAttention(num_units,encoder_output, in_seq_len, normalize = True)
    # attention_mechanim = tf.contrib.seq2seq.LuongAttention(num_units,encoder_output, in_seq_len, scale = True)

    #定义一个RNNCell实例；
    cell = getLayeredCell(layer_size, num_units, input_keep_prob)
    #利用AttentionWrapper进行封装，在多层rnn上加入了注意力机制。
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanim,attention_layer_size=num_units)
    return cell

#使用了一个全连接层计算decoder的输出
def decoder_projection(output, output_size):
    return tf.layers.dense(output, output_size, activation=None,use_bias=False, name='output_mlp')

def train_decoder(encoder_output, in_seq_len, target_seq, target_seq_len,encoder_state, num_units, layers, embedding, output_size,input_keep_prob, projection_layer):
    #attention_decoder_cell建立了上述的带注意力机制的RNN模型
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,layers, input_keep_prob)
    batch_size = tf.shape(in_seq_len)[0]
    # 将encoder的输出作为decoder的输入
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
    # 用来帮助decoder训练的helper
    helper = tf.contrib.seq2seq.TrainingHelper(target_seq, target_seq_len, time_major=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,init_state, output_layer=projection_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations=100)
    return outputs.rnn_output


def infer_decoder(encoder_output, in_seq_len, encoder_state, num_units, layers,embedding, output_size, input_keep_prob, projection_layer):
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,layers, input_keep_prob)

    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

    # TODO: start tokens and end tokens are hard code
    """
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, tf.fill([batch_size], 0), 1)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
            init_state, output_layer=projection_layer)
    """

    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding,
        start_tokens=tf.fill([batch_size], 0),
        end_token=1,
        initial_state=init_state,
        beam_width=10,
        output_layer=projection_layer,
        length_penalty_weight=1.0)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations=100)
    return outputs.sample_id

#序列到序列模型
#embedding向量----encoder层----带attention机制的decoder层
def seq2seq(in_seq, in_seq_len, target_seq, target_seq_len, vocab_size,num_units, layers, dropout):
    in_shape = tf.shape(in_seq)
    batch_size = in_shape[0]

    if target_seq != None:
        input_keep_prob = 1 - dropout
    else:
        input_keep_prob = 1

    projection_layer=layers_core.Dense(vocab_size, use_bias=False)

    #对输入和输出序列做embedding
    with tf.device('/cpu:0'):
        embedding = tf.get_variable(
                name = 'embedding',
                shape = [vocab_size, num_units])
    embed_input = tf.nn.embedding_lookup(embedding, in_seq, name='embed_input')

    # 编码
    encoder_output, encoder_state = bi_encoder(embed_input, in_seq_len,num_units, layers, input_keep_prob)


    #解码
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,layers, input_keep_prob)
    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

    if target_seq != None:
        embed_target = tf.nn.embedding_lookup(embedding, target_seq,name='embed_target')
        helper = tf.contrib.seq2seq.TrainingHelper(embed_target, target_seq_len, time_major=False)
    else:
        # TODO: start tokens and end tokens are hard code
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([batch_size], 0), 1)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,init_state, output_layer=projection_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations=100)
    if target_seq != None:
        return outputs.rnn_output
    else:
        return outputs.sample_id


def seq_loss(output, target, seq_len):
    target = target[:, 1:]
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=target)
    batch_size = tf.shape(target)[0]
    loss_mask = tf.sequence_mask(seq_len, tf.shape(output)[1])
    cost = cost * tf.to_float(loss_mask)
    return tf.reduce_sum(cost) / tf.to_float(batch_size)

