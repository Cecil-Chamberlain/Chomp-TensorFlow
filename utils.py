import os
import pickle
import numpy as np


class DataLoader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        in_vocab_txt = os.path.join(data_dir, "in_vocab.txt")
        out_vocab_txt = os.path.join(data_dir, "out_vocab.txt")
        input_files = os.listdir(os.path.join(data_dir, "melody/"))
        output_files = os.listdir(os.path.join(data_dir, "chords/"))
        in_vocab_file = os.path.join(data_dir, "in_vocab.pkl")
        out_vocab_file = os.path.join(data_dir, "out_vocab.pkl")
        input_dataset_file = os.path.join(data_dir, "input_data.npy")
        output_dataset_file = os.path.join(data_dir, "output_data.npy")

        if not (os.path.exists(in_vocab_file) and os.path.exists(input_dataset_file)):
            print("preprocessing dataset")
            self.preprocess(in_vocab_txt, out_vocab_txt, input_files, output_files, in_vocab_file, out_vocab_file, input_dataset_file, output_dataset_file)
        else:
            print("loading preprocessed dataset")
            self.load_preprocessed(in_vocab_file, out_vocab_file, input_dataset_file, output_dataset_file)
        self.create_batches()
        self.reset_batch_pointers()

    def preprocess(self, in_vocab_txt, out_vocab_txt, input_files, output_files, in_vocab_file, out_vocab_file, input_dataset_file, output_dataset_file):
        # in vocab
        with open(in_vocab_txt, "r") as f:
            in_voc = [word for line in f for word in line.split()]
        self.in_chars = in_voc
        self.in_vocab_size = len(self.in_chars)
        self.in_vocab = {key: val for val, key in enumerate(self.in_chars)}
        with open(in_vocab_file, 'wb') as f:
            pickle.dump(self.in_vocab, f)
        
        # out vocab
        with open(out_vocab_txt, "r") as f:
            out_voc = [word for line in f for word in line.split()]
        self.out_chars = out_voc
        self.out_vocab_size = len(self.out_chars)
        self.out_vocab = {key: val for val, key in enumerate(self.out_chars)}
        with open(out_vocab_file, 'wb') as f:
            pickle.dump(self.out_vocab, f)

        # dataset transposition to enlarge the dataset
        self.input_dataset = []
        self.output_dataset = []

        self.num_chords = int((len(self.out_vocab) - 1) / 12)

        for file in input_files:

            # open input data
            with open(self.data_dir + '/melody/' + file, "r") as f:
                in_data = [word for line in f for word in line.split()]

            # verify data is in vocab
            for index, key in enumerate(in_data):
                assert str(key) in self.in_vocab, "WARNING: Note '{}' in file '{}' not in vocabulary".format(in_data[index],file)

            in_data = np.array(list(map(self.in_vocab.get, in_data)))
            assert len(in_data) % 4 == 0, "Num events in '{}' is not multiple of 4".format(file)


            # calculate note transposition

            if np.amax(in_data) > 90:
                transpose_up = 1
            else:
                transpose_up = 13 # from 0 transposition to a perfect 5th
            
            if np.amin(in_data[np.nonzero(in_data)]) <  51:
                transpose_down = 1
            else:
                transpose_down = 13 # from 0 transposition to a 5th below
            

            # open output data
            with open(self.data_dir + '/chords/' + file, "r") as f:
                out_data = [word for line in f for word in line.split()]
            
            for index, key in enumerate(out_data):
                assert key in self.out_vocab, "WARNING: Note '{}' in file '{}' not in vocabulary".format(out_data[index],file)

            out_data = np.array(list(map(self.out_vocab.get, out_data)))
            assert len(in_data) % 4 == 0, "Num events in '{}' is not multiple of 4".format(file)
            
            # check input file is 4x output file
            assert len(in_data) == len(out_data) * 4, "WARNING: '{}' files are incompatible, input must have 4x the note events as chords but are {} and {}".format(file,len(in_data),len(out_data))
    
            # transpose up
            for semi_tones in range(transpose_up):
                in_transpose_up = np.array(in_data)
                out_transpose_up = np.array(out_data)
                mask = in_transpose_up > 0
                for ind, val in enumerate(out_transpose_up):
                    if val != 0:
                        y = np.zeros((self.num_chords * 12))
                        y[val-1] = 1
                        y = np.roll(y, semi_tones*self.num_chords, axis=0)
                        y = np.argmax(y)
                        out_transpose_up[ind] = y+1
                in_transpose_up[mask] = in_transpose_up[mask] + semi_tones
                self.input_dataset.append(in_transpose_up)
                self.output_dataset.append(out_transpose_up)

            # transpose down
            for semi_tones in range(transpose_down):
                in_transpose_down = np.array(in_data)
                mask = in_transpose_down > 0
                out_transpose_down = np.array(out_data)
                for ind, val in enumerate(out_transpose_down):
                    if val != 0:
                        y = np.zeros((self.num_chords * 12))
                        y[val-1] = 1
                        y = np.roll(y, -(semi_tones*self.num_chords), axis=0)
                        y = np.argmax(y)
                        out_transpose_down[ind] = y+1
                in_transpose_down[mask] = in_transpose_down[mask] - semi_tones
                self.input_dataset.append(in_transpose_down)
                self.output_dataset.append(out_transpose_down)

        # Export transposed dataset to one file
        np.save(input_dataset_file, self.input_dataset)
        np.save(output_dataset_file, self.output_dataset)
        self.input_dataset = np.array(self.input_dataset)
        self.output_dataset = np.array(self.output_dataset)

    def load_preprocessed(self, in_vocab_file, out_vocab_file, input_dataset_file, output_dataset_file):

        with open(in_vocab_file, 'rb') as f:
            self.in_vocab = pickle.load(f)
        self.in_vocab_size = len(self.in_vocab)
        self.input_dataset = np.load(input_dataset_file)

        with open(out_vocab_file, 'rb') as f:
            self.out_vocab = pickle.load(f)
        self.out_vocab_size = len(self.out_vocab)
        self.output_dataset = np.load(output_dataset_file)

    def create_batches(self):
        
        xdata = self.input_dataset
        ydata = self.output_dataset

        data_len = len(self.input_dataset)

        shuffle = np.random.permutation(data_len)
        division = int(data_len * 0.8)
        train_idx, test_idx = shuffle[:division], shuffle[division:]
        xtrain, xtest = np.concatenate(xdata[train_idx]), np.concatenate(xdata[test_idx])
        ytrain, ytest = np.concatenate(ydata[train_idx]), np.concatenate(ydata[test_idx])

        # clip ends off dataset to ensure they fit the batch size and sequence length
        xtrc = -int(len(xtrain) % (self.batch_size * self.seq_length)) if len(xtrain) % (self.batch_size * self.seq_length) != 0 else len(xtrain)
        xtec = -int(len(xtest) % (self.batch_size * self.seq_length)) if len(xtest) % (self.batch_size * self.seq_length) != 0 else len(xtest)
        ytrc = -int(len(ytrain) % (self.batch_size * self.seq_length / 4)) if int(len(ytrain) % (self.batch_size * self.seq_length / 4)) != 0 else len(ytrain)
        ytec = -int(len(ytest) % (self.batch_size * self.seq_length / 4)) if int(len(ytest) % (self.batch_size * self.seq_length / 4)) != 0 else len(ytest)
        

        self.x_train, self.x_test = np.reshape(xtrain[:xtrc], (self.batch_size, -1)), np.reshape(xtest[:xtec], (self.batch_size,-1))
        self.y_train, self.y_test = np.reshape(ytrain[:ytrc], (self.batch_size,-1)), np.reshape(ytest[:ytec], (self.batch_size,-1))
        self.in_num_batches = int(len(self.x_train[0]) / self.seq_length)
        self.out_num_batches = self.seq_length
        self.test_num_batches = int(len(self.x_test[0]) / self.seq_length)


    def next_train_batch(self):
        x_train, y_train = self.x_train[:,self.train_in_from:self.train_in_to], self.y_train[:,self.train_out_from:self.train_out_to]
        self.train_in_from += self.seq_length
        self.train_in_to += self.seq_length
        self.train_out_from += int(self.seq_length / 4)
        self.train_out_to += int(self.seq_length / 4)

        return x_train, y_train

    def next_test_batch(self):
        x_test, y_test = self.x_test[:,self.test_in_from:self.test_in_to], self.y_test[:,self.test_out_from:self.test_out_to]
        self.test_in_from += self.seq_length
        self.test_in_to += self.seq_length
        self.test_out_from += int(self.seq_length / 4)
        self.test_out_to += int(self.seq_length / 4)

        return x_test, y_test

    def reset_batch_pointers(self):
        self.train_in_from = 0
        self.train_in_to = self.train_in_from + self.seq_length
        self.train_out_from = 1
        self.train_out_to = self.train_out_from + int(self.seq_length / 4)
        self.test_in_from = 0
        self.test_in_to = self.test_in_from + self.seq_length
        self.test_out_from = 1
        self.test_out_to = self.test_out_from + int(self.seq_length / 4)

