'''Data utilities'''
import threading
import random
import numpy as np
from python_speech_features.base import fbank, delta
import tensorflow as tf
from deepsphinx.vocab import VOCAB_TO_INT
from deepsphinx.utils import FLAGS
from deepsphinx.fst import in_fst
import soundfile as sf
import csv

def get_features(audio_file):
    '''Get features from a file'''
    signal, sample_rate = sf.read(tf.gfile.FastGFile(audio_file, 'rb'))
    feat, energy = fbank(signal, sample_rate, nfilt=FLAGS.nfilt)
    feat = np.log(feat)
    dfeat = delta(feat, 2)
    ddfeat = delta(dfeat, 2)
    return np.concatenate([feat, dfeat, ddfeat, np.expand_dims(energy, 1)],
                          axis=1)

def get_speaker_stats(set_ids):
    '''Get mean and variance of a speaker'''
    tf.logging.info('Getting speaker stats')
    trans1 = tf.gfile.FastGFile(FLAGS.source_file_1).readlines()
    trans2 = tf.gfile.FastGFile(FLAGS.dann_file).readlines()
    trans = trans1 + trans2
    sum_speaker = {}
    sum_sq_speaker = {}
    count_speaker = {}
    for _, set_id, speaker, audio_file in csv.reader(trans):
        if set_id in set_ids:
            n_feat = 3 * FLAGS.nfilt + 1
            if speaker not in sum_speaker:
                sum_speaker[speaker] = np.zeros(n_feat)
                sum_sq_speaker[speaker] = np.zeros(n_feat)
                count_speaker[speaker] = 0
            feat = get_features(audio_file)
            sum_speaker[speaker] += np.mean(feat, 0)
            sum_sq_speaker[speaker] += np.mean(np.square(feat), 0)
            count_speaker[speaker] += 1
    mean = {k: sum_speaker[k] / count_speaker[k] for k, v in sum_speaker.items()}
    var = {k: sum_sq_speaker[k] / count_speaker[k] -
              np.square(mean[k]) for k, v in sum_speaker.items()}
#    print(mean)
#    print(var)
    return mean, var


def read_data_queue(
        set_id,
        queue1,
        queue2,
        queue3,
        queue4,
        queue_,
        sess,
        mean_speaker,
        var_speaker,
        fst):
    '''Start a thread to add data in a queue'''
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.nfilt * 3 + 1])
    input_length = tf.placeholder(dtype=tf.int32, shape=[])
    output_data = tf.placeholder(dtype=tf.int32, shape=[None])
    output_length = tf.placeholder(dtype=tf.int32, shape=[])
    enqueue_op1 = queue1.enqueue(
        [input_data, input_length, output_data, output_length])
    enqueue_op2 = queue2.enqueue(
        [input_data, input_length, output_data, output_length])
    enqueue_op3 = queue3.enqueue(
        [input_data, input_length, output_data, output_length])
    enqueue_op4 = queue4.enqueue(
        [input_data, input_length, output_data, output_length])
    #enqueue_op5 = queue5.enqueue(
    #    [input_data, input_length, output_data, output_length])
    #enqueue_op6 = queue6.enqueue(
    #    [input_data, input_length, output_data, output_length])
    enqueue_op_ = queue_.enqueue(
        [input_data, input_length, output_data, output_length])
    
    close_op1 = queue1.close()
    close_op2 = queue2.close()
    close_op3 = queue3.close()
    close_op4 = queue4.close()
    #close_op5 = queue5.close()
    #close_op6 = queue6.close()
    close_op_ = queue_.close()

    thread1 = threading.Thread(
        target=read_data_thread1,
        args=(
            set_id,
            sess,
            input_data,
            input_length,
            output_data,
            output_length,
            enqueue_op1,
            close_op1,
            mean_speaker,
            var_speaker,
            fst))
    thread1.daemon = True  # Thread will close when parent quits.
#    thread1.start()
    
    thread2 = threading.Thread(
        target=read_data_thread2,
        args=(
            set_id,
            sess,
            input_data,
            input_length,
            output_data,
            output_length,
            enqueue_op2,
            close_op2,
            mean_speaker,
            var_speaker,
            fst))
    thread2.daemon = True  # Thread will close when parent quits.

    thread3 = threading.Thread(
        target=read_data_thread3,
        args=(
            set_id,
            sess,
            input_data,
            input_length,
            output_data,
            output_length,
            enqueue_op3,
            close_op3,
            mean_speaker,
            var_speaker,
            fst))
    thread3.daemon = True  # Thread will close when parent quits.

    thread4 = threading.Thread(
        target=read_data_thread4,
        args=(
            set_id,
            sess,
            input_data,
            input_length,
            output_data,
            output_length,
            enqueue_op4,
            close_op4,
            mean_speaker,
            var_speaker,
            fst))
    thread4.daemon = True  # Thread will close when parent quits.

    """
    thread5 = threading.Thread(
        target=read_data_thread5,
        args=(
            set_id,
            sess,
            input_data,
            input_length,
            output_data,
            output_length,
            enqueue_op5,
            close_op5,
            mean_speaker,
            var_speaker,
            fst))
    thread5.daemon = True  # Thread will close when parent quits.

    thread6 = threading.Thread(
        target=read_data_thread6,
        args=(
            set_id,
            sess,
            input_data,
            input_length,
            output_data,
            output_length,
            enqueue_op6,
            close_op6,
            mean_speaker,
            var_speaker,
            fst))
    thread6.daemon = True  # Thread will close when parent quits.
    """

    thread_ = threading.Thread(
        target=read_data_thread_,
        args=(
            set_id,
            sess,
            input_data,
            input_length,
            output_data,
            output_length,
            enqueue_op_,
            close_op_,
            mean_speaker,
            var_speaker,
            fst))
    thread_.daemon = True  # Thread will close when parent quits.

    thread1.start()   
    thread2.start()
    thread3.start()
    thread4.start()
    #thread5.start()
    #thread6.start()
    thread_.start()



def read_data_thread1(
        set_id,
        sess,
        input_data,
        input_length,
        output_data,
        output_length,
        enqueue_op1,
        close_op1,
        mean_speaker,
        var_speaker,
        fst):
    '''Enqueue data to queue'''

    trans = tf.gfile.FastGFile(FLAGS.source_file_1).readlines()
    random.shuffle(trans)
    for text, set_id_trans, speaker, audio_file in csv.reader(trans):
        try:
            text = [VOCAB_TO_INT[c]
                    for c in list(text)] + [VOCAB_TO_INT['</s>']]
     #       tf.logging.info('0000000000000000000')
        except KeyError:
            continue
     #       tf.logging.info('111111111111111111111')
        if (set_id == set_id_trans and
                ((not FLAGS.use_train_lm) or in_fst(fst, text))):
            feat = get_features(audio_file)
            feat = feat - mean_speaker[speaker]
            feat = feat / np.sqrt(var_speaker[speaker])
            sess.run(enqueue_op1, feed_dict={
                input_data: feat,
                input_length: feat.shape[0],
                output_data: text,
                output_length: len(text)})
    sess.run(close_op1)

def read_data_thread2(
        set_id,
        sess,
        input_data,
        input_length,
        output_data,
        output_length,
        enqueue_op2,
        close_op2,
        mean_speaker,
        var_speaker,
        fst):
    '''Enqueue data to queue'''

    trans = tf.gfile.FastGFile(FLAGS.source_file_2).readlines()
    random.shuffle(trans)
    for text, set_id_trans, speaker, audio_file in csv.reader(trans):
        try:
            text = [VOCAB_TO_INT[c]
                    for c in list(text)] + [VOCAB_TO_INT['</s>']]
     #       tf.logging.info('0000000000000000000')
        except KeyError:
            continue
     #       tf.logging.info('111111111111111111111')
        if (set_id == set_id_trans and
                ((not FLAGS.use_train_lm) or in_fst(fst, text))):
            feat = get_features(audio_file)
            feat = feat - mean_speaker[speaker]
            feat = feat / np.sqrt(var_speaker[speaker])
            sess.run(enqueue_op2, feed_dict={
                input_data: feat,
                input_length: feat.shape[0],
                output_data: text,
                output_length: len(text)})
    sess.run(close_op2)

def read_data_thread3(
        set_id,
        sess,
        input_data,
        input_length,
        output_data,
        output_length,
        enqueue_op3,
        close_op3,
        mean_speaker,
        var_speaker,
        fst):
    '''Enqueue data to queue'''

    trans = tf.gfile.FastGFile(FLAGS.source_file_3).readlines()
    random.shuffle(trans)
    for text, set_id_trans, speaker, audio_file in csv.reader(trans):
        try:
            text = [VOCAB_TO_INT[c]
                    for c in list(text)] + [VOCAB_TO_INT['</s>']]
     #       tf.logging.info('0000000000000000000')
        except KeyError:
            continue
     #       tf.logging.info('111111111111111111111')
        if (set_id == set_id_trans and
                ((not FLAGS.use_train_lm) or in_fst(fst, text))):
            feat = get_features(audio_file)
            feat = feat - mean_speaker[speaker]
            feat = feat / np.sqrt(var_speaker[speaker])
            sess.run(enqueue_op3, feed_dict={
                input_data: feat,
                input_length: feat.shape[0],
                output_data: text,
                output_length: len(text)})
    sess.run(close_op3)

def read_data_thread4(
        set_id,
        sess,
        input_data,
        input_length,
        output_data,
        output_length,
        enqueue_op4,
        close_op4,
        mean_speaker,
        var_speaker,
        fst):
    '''Enqueue data to queue'''

    trans = tf.gfile.FastGFile(FLAGS.source_file_4).readlines()
    random.shuffle(trans)
    for text, set_id_trans, speaker, audio_file in csv.reader(trans):
        try:
            text = [VOCAB_TO_INT[c]
                    for c in list(text)] + [VOCAB_TO_INT['</s>']]
     #       tf.logging.info('0000000000000000000')
        except KeyError:
            continue
     #       tf.logging.info('111111111111111111111')
        if (set_id == set_id_trans and
                ((not FLAGS.use_train_lm) or in_fst(fst, text))):
            feat = get_features(audio_file)
            feat = feat - mean_speaker[speaker]
            feat = feat / np.sqrt(var_speaker[speaker])
            sess.run(enqueue_op4, feed_dict={
                input_data: feat,
                input_length: feat.shape[0],
                output_data: text,
                output_length: len(text)})
    sess.run(close_op4)

"""
def read_data_thread5(
        set_id,
        sess,
        input_data,
        input_length,
        output_data,
        output_length,
        enqueue_op5,
        close_op5,
        mean_speaker,
        var_speaker,
        fst):
    '''Enqueue data to queue'''

    trans = tf.gfile.FastGFile(FLAGS.source_file_5).readlines()
    random.shuffle(trans)
    for text, set_id_trans, speaker, audio_file in csv.reader(trans):
        try:
            text = [VOCAB_TO_INT[c]
                    for c in list(text)] + [VOCAB_TO_INT['</s>']]
     #       tf.logging.info('0000000000000000000')
        except KeyError:
            continue
     #       tf.logging.info('111111111111111111111')
        if (set_id == set_id_trans and
                ((not FLAGS.use_train_lm) or in_fst(fst, text))):
            feat = get_features(audio_file)
            feat = feat - mean_speaker[speaker]
            feat = feat / np.sqrt(var_speaker[speaker])
            sess.run(enqueue_op5, feed_dict={
                input_data: feat,
                input_length: feat.shape[0],
                output_data: text,
                output_length: len(text)})
    sess.run(close_op5)

def read_data_thread6(
        set_id,
        sess,
        input_data,
        input_length,
        output_data,
        output_length,
        enqueue_op6,
        close_op6,
        mean_speaker,
        var_speaker,
        fst):
    '''Enqueue data to queue'''

    trans = tf.gfile.FastGFile(FLAGS.source_file_6).readlines()
    random.shuffle(trans)
    for text, set_id_trans, speaker, audio_file in csv.reader(trans):
        try:
            text = [VOCAB_TO_INT[c]
                    for c in list(text)] + [VOCAB_TO_INT['</s>']]
     #       tf.logging.info('0000000000000000000')
        except KeyError:
            continue
     #       tf.logging.info('111111111111111111111')
        if (set_id == set_id_trans and
                ((not FLAGS.use_train_lm) or in_fst(fst, text))):
            feat = get_features(audio_file)
            feat = feat - mean_speaker[speaker]
            feat = feat / np.sqrt(var_speaker[speaker])
            sess.run(enqueue_op6, feed_dict={
                input_data: feat,
                input_length: feat.shape[0],
                output_data: text,
                output_length: len(text)})
    sess.run(close_op6)
"""

"""
    trans_ = tf.gfile.FastGFile(FLAGS.dann_file).readlines()
    random.shuffle(trans_)
    for text, set_id_trans, speaker, audio_file in csv.reader(trans_):
        try:
            text = [VOCAB_TO_INT[c]
                    for c in list(text)] + [VOCAB_TO_INT['</s>']]
     #       tf.logging.info('0000000000000000000')
        except KeyError:
            continue
     #       tf.logging.info('111111111111111111111')
        if (set_id == set_id_trans and
                ((not FLAGS.use_train_lm) or in_fst(fst, text))):
            feat = get_features(audio_file)
            feat = feat - mean_speaker[speaker]
            feat = feat / np.sqrt(var_speaker[speaker])
            sess.run(enqueue_op_, feed_dict={
                input_data: feat,
                input_length: feat.shape[0],
                output_data: text,
                output_length: len(text)})
    
    sess.run(close_op)
    sess.run(close_op_)
"""

def read_data_thread_(
        set_id,
        sess,
        input_data,
        input_length,
        output_data,
        output_length,
        enqueue_op_,
        close_op_,
        mean_speaker,
        var_speaker,
        fst):
    '''Enqueue data to queue'''

    trans = tf.gfile.FastGFile(FLAGS.dann_file).readlines()
    random.shuffle(trans)
    for text, set_id_trans, speaker, audio_file in csv.reader(trans):
        try:
            text = [VOCAB_TO_INT[c]
                    for c in list(text)] + [VOCAB_TO_INT['</s>']]
     #       tf.logging.info('0000000000000000000')
        except KeyError:
            continue
     #       tf.logging.info('111111111111111111111')
        if (set_id == set_id_trans and
                ((not FLAGS.use_train_lm) or in_fst(fst, text))):
            feat = get_features(audio_file)
            feat = feat - mean_speaker[speaker]
            feat = feat / np.sqrt(var_speaker[speaker])
            sess.run(enqueue_op_, feed_dict={
                input_data: feat,
                input_length: feat.shape[0],
                output_data: text,
                output_length: len(text)})
    sess.run(close_op_)

