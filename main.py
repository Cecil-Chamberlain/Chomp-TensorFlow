from __future__ import print_function
import tensorflow as tf
import numpy as np

import argparse
import os
import pickle

from utils import DataLoader
from model import Model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', action='store', type=str, default='run')

    parser.add_argument('--from_file', action='store', type=bool, default=False)

    parser.add_argument('--data_dir', action='store', type=str, default='dataset')
    
    parser.add_argument('--save_dir', action='store', type=str, default='save')

    parser.add_argument('--num_units', action='store', type=int, default=128)
    
    parser.add_argument('--num_layers', action='store', type=int, default=1)
    
    parser.add_argument('--batch_size', action='store', type=int, default=10)

    parser.add_argument('--seq_length', action='store', type=int, default=128)
    
    parser.add_argument('--num_epochs', action='store', type=int, default=40)
    
    parser.add_argument('--grad_clip', action='store', type=float, default=5.0)
    
    parser.add_argument('--learning_rate', action='store', type=float, default=0.002)

    parser.add_argument('--input_keep_prob', action='store', type=float, default=1)
    
    parser.add_argument('--output_keep_prob', action='store', type=float, default=1)
    

    args = parser.parse_args()
    mode = globals()[args.mode]
    mode(args)


def train(args):
    data_loader = DataLoader(args.data_dir, args.batch_size, args.seq_length)
    args.in_vocab_size = data_loader.in_vocab_size
    args.out_vocab_size = data_loader.out_vocab_size
    
    model = Model(args)
    

    with tf.Session() as sess:


        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(args.num_epochs):
            data_loader.reset_batch_pointers()
            state = sess.run(model.initial_state)
            if epoch > 0:
                print("Epoch: {}  Training loss avrg: {}  Testing loss avrg: {}"
                      .format(epoch,
                              round(t_loss/data_loader.in_num_batches, 4),
                              round(b_loss/data_loader.test_num_batches, 4)))
            else:
                print("Training started...")
            t_loss = 0
            b_loss = 0
            for batch in range(data_loader.in_num_batches-1):

                x, y = data_loader.next_train_batch()
                feed_dict = {model.input_data: x, model.targets: y, model.initial_state: state}

                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict=feed_dict)

                t_loss += np.mean(train_loss)

                if batch % int(data_loader.in_num_batches / data_loader.test_num_batches + 2) == 0:
                    x, y = data_loader.next_test_batch()
                    test_state = sess.run(model.cell.zero_state(args.batch_size, dtype=tf.float32))
                    feed_dict = {model.input_data: x, model.targets: y, model.initial_state: test_state}
                    test_loss = sess.run([model.cost], feed_dict=feed_dict)
                    b_loss += np.mean(test_loss)
                    
            saver.save(sess, args.save_dir+'/model.ckpt', global_step=epoch * data_loader.in_num_batches + batch)


def run(args):
    try:
        with open(os.path.join(args.data_dir, 'in_vocab.pkl'), 'rb') as f:
            in_vocab = pickle.load(f)
            args.in_vocab_size = len(in_vocab)
        with open(os.path.join(args.data_dir, 'out_vocab.pkl'), 'rb') as f:
            out_vocab = pickle.load(f)
            args.out_vocab_size = len(out_vocab)
    except FileNotFoundError:
        print("No vocab files found, ensure to run 'main.py --mode train' first")

    model = Model(args, training=False)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(args.save_dir))
        model.generate_chords(sess, in_vocab, out_vocab, args.from_file)


if __name__ == '__main__':
    main()
