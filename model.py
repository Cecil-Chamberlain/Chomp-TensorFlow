import tensorflow as tf
from midi_interface import MIDIDevice

import time
import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 4

        cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(args.num_units),
                               input_keep_prob=args.input_keep_prob,
                               output_keep_prob=args.output_keep_prob)

        self.cell = cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(args.num_layers)], state_is_tuple=True)

        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, int(args.seq_length / 4)])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)


        softmax_w = tf.Variable(np.random.rand(args.num_units, args.out_vocab_size), dtype=tf.float32)
        softmax_b = tf.Variable(np.zeros((args.out_vocab_size)), dtype=tf.float32)

        inputs = tf.contrib.layers.embed_sequence(self.input_data, args.in_vocab_size, args.num_units)

        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self.initial_state)

        self.final_state = last_state
        output = tf.reshape(outputs, [-1, args.num_units])
        
        self.projection = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.projection)
        
        if training:
            
            self.logits = tf.reshape(self.projection, [args.batch_size, int(args.seq_length / 4), 4, args.out_vocab_size])
            self.logits = tf.reshape(self.logits[::,:,-1], [-1, args.out_vocab_size])
            self.labels = tf.reshape(self.targets, [-1])

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.labels)

            self.cost = tf.reduce_sum(loss) / args.batch_size / int(args.seq_length / 4)
            
            self.learning_rate = args.learning_rate
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                    args.grad_clip)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def generate_chords(self, sess, in_vocab, out_vocab, from_file=False):
        state = sess.run(self.cell.zero_state(1, tf.float32))

        rev_out_vocab = {key:val for val, key in out_vocab.items()}

        device = MIDIDevice()


        with open('dataset/testing/melody/misty', "r") as f:
                file = [word for line in f for word in line.split()]
        testing = np.array(list(map(in_vocab.get, file)))
        index = 0

        output = []

        while True:
            x = np.zeros((1, 4))
            if from_file:
                try:
                    x[0] = testing[index:index+4]
                    index += 4
                except:
                    break
            else:
                for i in range(4):
                    x[0, i] = in_vocab[device.receive()]
                    time.sleep(1/4)
            feed_dict = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed_dict=feed_dict)
            p = probs[0]

            sample = np.argmax(p)

            pred = rev_out_vocab[sample]
            print("pred: ", pred)
            output.append(pred)
        with open("dataset/testing/misty output", "w") as f:
            for i, chord in enumerate(output):
                if i % 4 == 0:
                    f.write('\n')
                f.write(chord + ' ')
                

            
