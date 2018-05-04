'''Tensorflow model for speech recognition'''
import tensorflow as tf
import numpy as np
from flip_gradient import flip_gradient
from deepsphinx.vocab import VOCAB_SIZE, VOCAB_TO_INT
from deepsphinx.utils import FLAGS
from deepsphinx.lm import LMCellWrapper
from deepsphinx.attention import BahdanauAttentionCutoff
from deepsphinx.beam_search_decoder import BeamSearchDecoder

tf.contrib.seq2seq.BeamSearchDecoder = BeamSearchDecoder


def encoding_layer(
        input_lengths,
        rnn_inputs,
        keep_prob):
    ''' Encoding layer for the model.

    Args:
        input_lengths (Tensor): A tensor of input lenghts of instances in
            batches
        rnn_inputs (Tensor): Inputs

    Returns:
        Encoding output, LSTM state, output length
    '''
#    with tf.variable_scope("foo",reuse=tf.AUTO_REUSE)
    for layer in range(FLAGS.num_conv_layers):
        with tf.variable_scope("foo_{}".format(layer), reuse=tf.AUTO_REUSE):
           filter = tf.get_variable(
               "conv_filter{}".format(layer + 1),
                shape=[FLAGS.conv_layer_width, rnn_inputs.get_shape()[2], FLAGS.conv_layer_size])
           rnn_inputs = tf.nn.conv1d(rnn_inputs, filter, 1, 'SAME')
           #scope.reuse_variables()
    for layer in range(FLAGS.num_layers):
        with tf.variable_scope('encoder_{}'.format(layer), reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(
                FLAGS.rnn_size,
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell_fw,
                output_keep_prob=keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=rnn_inputs.get_shape()[2])

            cell_bw = tf.contrib.rnn.LSTMCell(
                FLAGS.rnn_size,
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw,
                output_keep_prob=keep_prob,
                variational_recurrent=True,
                dtype=tf.float32,
                input_size=rnn_inputs.get_shape()[2])

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                rnn_inputs,
                input_lengths,
                dtype=tf.float32)

            if layer != FLAGS.num_layers - 1:
                rnn_inputs = tf.concat(enc_output,2)
                rnn_inputs = rnn_inputs[:, ::2, :]
                input_lengths = (input_lengths + 1) // 2
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output, 2)
    
    print(enc_output.get_shape())

    return enc_output, enc_state, input_lengths


def dom_class_cell_1(
        enc_output,
        enc_output_lengths):

    '''Domain Classifier1'''  

    with tf.variable_scope("domclass_1",reuse = tf.AUTO_REUSE):
    
       num_hidden = 24
       #cell_1 = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
       #cell_2 = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)

       def create_cell():
           cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
           return cell

       cell_1 = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(2)], state_is_tuple=True)
       cell_2 = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(2)], state_is_tuple=True)

       (val1, val2), state = tf.nn.bidirectional_dynamic_rnn(cell_1,cell_2,inputs=enc_output,dtype=tf.float32,sequence_length=enc_output_lengths)

       val = tf.concat([val1,val2],axis=2)
       val = tf.transpose(val,[1,0,2])
       print(val.get_shape())
       last = tf.gather(val,tf.shape(val)[0] - 1)
       #print(last.get_shape())
       w1 = tf.get_variable("w1",[2*num_hidden,1024],initializer=tf.random_normal_initializer())

       b1 = tf.get_variable("b1",[1024],initializer=tf.constant_initializer(0.1))
        
       w2 = tf.get_variable("w2",[1024,2],initializer=tf.random_normal_initializer(0.1))
    
       b2 = tf.get_variable("b2",[2],initializer=tf.constant_initializer(0.1))
        
       temp = tf.matmul(last,w1) + b1
       prediction =  tf.nn.softmax(tf.matmul(temp,w2) + b2)
       
    return prediction    

def dom_class_cell_2(
        enc_output,
        enc_output_lengths):

    '''Domain Classifier2'''  


    with tf.variable_scope("domclass_2",reuse = tf.AUTO_REUSE):
    
       num_hidden = 24
       #cell_1 = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
       #cell_2 = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)

       def create_cell():
           cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
           return cell

       cell_1 = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(2)], state_is_tuple=True)
       cell_2 = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(2)], state_is_tuple=True)


       (val1, val2) ,state = tf.nn.bidirectional_dynamic_rnn(cell_1,cell_2,inputs=enc_output,dtype=tf.float32,sequence_length=enc_output_lengths)

       val = tf.concat([val1,val2],axis=2)
       val = tf.transpose(val,[1,0,2])
       print(val.get_shape())
       last = tf.gather(val,tf.shape(val)[0] - 1)
       #print(last.get_shape())
       w1 = tf.get_variable("w1",[2*num_hidden,1024],initializer=tf.random_normal_initializer())

       b1 = tf.get_variable("b1",[1024],initializer=tf.constant_initializer(0.1))
        
       w2 = tf.get_variable("w2",[1024,2],initializer=tf.random_normal_initializer(0.1))
    
       b2 = tf.get_variable("b2",[2],initializer=tf.constant_initializer(0.1))
        
       temp = tf.matmul(last,w1) + b1
       prediction =  tf.nn.softmax(tf.matmul(temp,w2) + b2)
       
    return prediction    

def dom_class_cell_3(
        enc_output,
        enc_output_lengths):

    '''Domain Classifier3'''  

    with tf.variable_scope("domclass_3",reuse = tf.AUTO_REUSE):
    
       num_hidden = 24
       #cell_1 = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
       #cell_2 = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)

       def create_cell():
           cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
           return cell

       cell_1 = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(2)], state_is_tuple=True)
       cell_2 = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(2)], state_is_tuple=True)


       (val1, val2), state = tf.nn.bidirectional_dynamic_rnn(cell_1,cell_2,inputs=enc_output,dtype=tf.float32,sequence_length=enc_output_lengths)

       val = tf.concat([val1,val2],axis=2)
       val = tf.transpose(val,[1,0,2])
       print(val.get_shape())
       last = tf.gather(val,tf.shape(val)[0] - 1)
       #print(last.get_shape())
       w1 = tf.get_variable("w1",[2*num_hidden,1024],initializer=tf.random_normal_initializer())

       b1 = tf.get_variable("b1",[1024],initializer=tf.constant_initializer(0.1))
        
       w2 = tf.get_variable("w2",[1024,2],initializer=tf.random_normal_initializer(0.1))
    
       b2 = tf.get_variable("b2",[2],initializer=tf.constant_initializer(0.1))
        
       temp = tf.matmul(last,w1) + b1
       prediction =  tf.nn.softmax(tf.matmul(temp,w2) + b2)
       
       
    return prediction    

def dom_class_cell_4(
        enc_output,
        enc_output_lengths):

    '''Domain Classifier4'''  

    with tf.variable_scope("domclass_4",reuse = tf.AUTO_REUSE):
    
       num_hidden = 24
       #cell_1 = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
       #cell_2 = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)

       def create_cell():
           cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
           return cell

       cell_1 = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(2)], state_is_tuple=True)
       cell_2 = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(2)], state_is_tuple=True)


       (val1, val2), state = tf.nn.bidirectional_dynamic_rnn(cell_1,cell_2,inputs=enc_output,dtype=tf.float32,sequence_length=enc_output_lengths)

       val = tf.concat([val1,val2],axis=2)
       val = tf.transpose(val,[1,0,2])
       print(val.get_shape())
       last = tf.gather(val,tf.shape(val)[0] - 1)
       #print(last.get_shape())
       w1 = tf.get_variable("w1",[2*num_hidden,1024],initializer=tf.random_normal_initializer())

       b1 = tf.get_variable("b1",[1024],initializer=tf.constant_initializer(0.1))
        
       w2 = tf.get_variable("w2",[1024,2],initializer=tf.random_normal_initializer(0.1))
    
       b2 = tf.get_variable("b2",[2],initializer=tf.constant_initializer(0.1))
        
       temp = tf.matmul(last,w1) + b1
       prediction =  tf.nn.softmax(tf.matmul(temp,w2) + b2)
       
       
    return prediction    


def get_dec_cell(
        enc_output,
        enc_output_lengths,
        use_lm,
        fst,
        tile_size,
        keep_prob):
    '''Decoding cell for attention based model

    Return:
        `RNNCell` Instance
    '''

    lstm = tf.contrib.rnn.LSTMCell(
        FLAGS.rnn_size,
        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    dec_cell_inp = tf.contrib.rnn.DropoutWrapper(
        lstm,
        output_keep_prob=keep_prob,
        variational_recurrent=True,
        dtype=tf.float32)
    lstm = tf.contrib.rnn.LSTMCell(
        FLAGS.rnn_size,
        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    dec_cell = tf.contrib.rnn.DropoutWrapper(
        lstm,
        output_keep_prob=keep_prob,
        variational_recurrent=True,
        dtype=tf.float32)

    dec_cell_out = tf.contrib.rnn.LSTMCell(
        FLAGS.rnn_size,
        num_proj=VOCAB_SIZE,
        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

    dec_cell = tf.contrib.rnn.MultiRNNCell(
        [dec_cell_inp] +
        [dec_cell] * (FLAGS.num_decoding_layers - 2) +
        [dec_cell_out])

    enc_output = tf.contrib.seq2seq.tile_batch(
        enc_output,
        tile_size)

    enc_output_lengths = tf.contrib.seq2seq.tile_batch(
        enc_output_lengths,
        tile_size)

    attn_mech = BahdanauAttentionCutoff(
        FLAGS.rnn_size,
        enc_output,
        enc_output_lengths,
        normalize=True,
        name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(
        dec_cell,
        attn_mech,
        VOCAB_SIZE,
        output_attention=True)

    if use_lm:
        dec_cell = LMCellWrapper(dec_cell, fst, 5)

    return dec_cell


#pylint: disable-msg=too-many-arguments
def training_decoding_layer(
        target_data,
        target_lengths,
        enc_output,
        enc_output_lengths,
        fst,
        keep_prob):
    ''' Training decoding layer for the model.

    Returns:
        Training logits
    '''
    target_data = tf.concat(
        [tf.fill([4*FLAGS.batch_size, 1], VOCAB_TO_INT['<s>']),
         target_data[:, :-1]], 1)

    dec_cell = get_dec_cell(
        enc_output,
        enc_output_lengths,
        FLAGS.use_train_lm,
        fst,
        1,
        keep_prob)

    initial_state = dec_cell.zero_state(
        dtype=tf.float32,
        batch_size=4*FLAGS.batch_size)

    target_data = tf.nn.embedding_lookup(
        tf.eye(VOCAB_SIZE),
        target_data)

    training_helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=target_data,
        sequence_length=target_lengths,
        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(
        dec_cell,
        training_helper,
        initial_state)

    training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(
        training_decoder,
        output_time_major=False,
        impute_finished=True)

    return training_logits


def inference_decoding_layer(
        enc_output,
        enc_output_lengths,
        fst,
        keep_prob):
    ''' Inference decoding layer for the model.

    Returns:
        Predictions
    '''

    dec_cell = get_dec_cell(
        enc_output,
        enc_output_lengths,
        FLAGS.use_inference_lm,
        fst,
        FLAGS.beam_width,
        keep_prob)

    initial_state = dec_cell.zero_state(
        dtype=tf.float32,
        batch_size=4*FLAGS.batch_size * FLAGS.beam_width)

    start_tokens = tf.fill(
        [4*FLAGS.batch_size],
        VOCAB_TO_INT['<s>'],
        name='start_tokens')

    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        dec_cell,
        tf.eye(VOCAB_SIZE),
        start_tokens,
        VOCAB_TO_INT['</s>'],
        initial_state,
        FLAGS.beam_width)

    predictions, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder,
        output_time_major=False,
        maximum_iterations=FLAGS.max_output_len)

    return predictions

def seq2seq_model(
        input_data,
        target_data,
        input_lengths,
        target_lengths,
        fst,
        keep_prob,
        training_mode,
        lparam):
    ''' Attention based model

    Returns:
        Logits, Predictions, Training operation, Cost, Step, Scores of beam
        search
    '''
      

    enc_output_, _, enc_lengths_ = encoding_layer(
        input_lengths,
        input_data,
        keep_prob)
    
    #Splitting the encoder outputs and combining them for input to domain classifiers

    enc_output_d = tf.slice(enc_output_,[4*FLAGS.batch_size,0,0],[FLAGS.batch_size,-1,-1])
    enc_lengths_d = tf.slice(enc_lengths_,[4*FLAGS.batch_size],[FLAGS.batch_size])


    enc_output1 = tf.slice(enc_output_, [0,0,0], [FLAGS.batch_size,-1,-1])
    enc_output1_t = tf.concat([enc_output1,enc_output_d],0)

    enc_output2 = tf.slice(enc_output_, [FLAGS.batch_size,0,0], [FLAGS.batch_size,-1,-1])
    enc_output2_t = tf.concat([enc_output2,enc_output_d],0)

    enc_output3 = tf.slice(enc_output_,[2 * FLAGS.batch_size,0,0],[FLAGS.batch_size,-1,-1])
    enc_output3_t = tf.concat([enc_output3,enc_output_d],0)

    enc_output4 = tf.slice(enc_output_,[3 * FLAGS.batch_size,0,0],[FLAGS.batch_size,-1,-1])
    enc_output4_t = tf.concat([enc_output4,enc_output_d],0)

    # Getting the encoded lengths for each domain and then combining them

    enc_lengths1 = tf.slice(enc_lengths_,[0],[FLAGS.batch_size])
    enc_lengths1_t = tf.concat([enc_lengths1,enc_lengths_d],0)

    enc_lengths2 = tf.slice(enc_lengths_,[FLAGS.batch_size],[FLAGS.batch_size])
    enc_lengths2_t = tf.concat([enc_lengths2,enc_lengths_d],0) 

    enc_lengths3 = tf.slice(enc_lengths_,[2*FLAGS.batch_size],[FLAGS.batch_size])
    enc_lengths3_t = tf.concat([enc_lengths3,enc_lengths_d],0)

    enc_lengths4 = tf.slice(enc_lengths_,[3*FLAGS.batch_size],[FLAGS.batch_size])
    enc_lengths4_t = tf.concat([enc_lengths4,enc_lengths_d],0)

    label_t1 = np.tile(np.array([1,0]),(FLAGS.batch_size,1))
    label_t2 = np.tile(np.array([0,1]),(FLAGS.batch_size,1))
    
    label_d = np.concatenate([label_t1,label_t2],axis=0)    
    
    label_domain = tf.convert_to_tensor(label_d)
    
    
    #Random Shuffle for domain classification
    '''
    label_domain = tf.random_shuffle(label_domain,seed=0)
    enc_output1_t = tf.random_shuffle(enc_output1_t,seed=0)
    enc_output2_t = tf.random_shuffle(enc_output2_t,seed=0)
    enc_output3_t = tf.random_shuffle(enc_output3_t,seed=0)
    enc_output4_t = tf.random_shuffle(enc_output4_t,seed=0)
    enc_lengths1_t = tf.random_shuffle(enc_lengths1_t,seed=0)
    enc_lengths2_t = tf.random_shuffle(enc_lengths2_t,seed=0)
    enc_lengths3_t = tf.random_shuffle(enc_lengths3_t,seed=0)
    enc_lengths4_t = tf.random_shuffle(enc_lengths4_t,seed=0)   
    '''
    
    #Training mode is set to True currently
     
    enc_output_t1 = lambda:tf.slice(enc_output_,[4*FLAGS.batch_size,0,0],[FLAGS.batch_size,-1,-1])
    enc_output_t2 = lambda: tf.slice(enc_output_,[0,0,0],[4*FLAGS.batch_size,-1,-1])
    
    enc_output_c = tf.cond(training_mode,enc_output_t2,enc_output_t1)

    enc_lengths_t1 = lambda:tf.slice(enc_lengths_,[4*FLAGS.batch_size],[FLAGS.batch_size])
    enc_lengths_t2 = lambda:tf.slice(enc_lengths_,[0],[4*FLAGS.batch_size])
    
    enc_lengths_c = tf.cond(training_mode,enc_lengths_t2,enc_lengths_t1)
    
    #Flip Gradient below is the gradient reversal layer
    
    grl1 = flip_gradient(enc_output1_t,lparam)
    grl2 = flip_gradient(enc_output2_t,lparam)
    grl3 = flip_gradient(enc_output3_t,lparam)
    grl4 = flip_gradient(enc_output4_t,lparam)
    
    #Predictions of the domain classifier
    
    predictions_1 = dom_class_cell_1(grl1,enc_lengths1_t)
    predictions_2 = dom_class_cell_2(grl2,enc_lengths2_t)
    predictions_3 = dom_class_cell_3(grl3,enc_lengths3_t)
    predictions_4 = dom_class_cell_4(grl4,enc_lengths4_t)
    
    #Domain classification loss

    loss_d1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
              labels = label_d,
              logits = predictions_1))
    
    loss_d2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
              labels = label_d,
              logits = predictions_2))
    
    loss_d3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
              labels = label_d,
              logits = predictions_3))
    
    loss_d4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
              labels = label_d,
              logits = predictions_4))
    
    #a = tf.shape(enc_output_c)
    #b = enc_lengths_c


    with tf.variable_scope('decode'):

        training_logits = training_decoding_layer(
            target_data,
            target_lengths,
            enc_output_c,
            enc_lengths_c,
            fst,
            keep_prob)
    with tf.variable_scope('decode', reuse=True):
        predictions = inference_decoding_layer(
            enc_output_c,
            enc_lengths_c,
            fst,
            keep_prob)

    # Create tensors for the training logits and predictions
    training_logits = tf.identity(
        training_logits.rnn_output,
        name='logits')
    scores = tf.identity(
        predictions.beam_search_decoder_output.scores,
        name='scores')
    predictions = tf.identity(
        predictions.predicted_ids,
        name='predictions')

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(
        target_lengths,
        tf.reduce_max(target_lengths),
        dtype=tf.float32,
        name='masks')

    with tf.name_scope('optimization'):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            target_data,
            masks)

        #Soft Optimization 
        loss_d =  tf.log(tf.exp(loss_d1)+tf.exp(loss_d2)+tf.exp(loss_d3)+tf.exp(loss_d4))

        

        cost = cost + loss_d
        tf.summary.scalar('cost', cost)

        step = tf.contrib.framework.get_or_create_global_step()

        # Optimizer
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [
            (tf.clip_by_value(grad, -5., 5.), var)
            for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, step)

    return training_logits, predictions, train_op, cost, step, scores, loss_d
