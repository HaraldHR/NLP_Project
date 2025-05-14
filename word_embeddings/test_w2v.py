import argparse
import json
import os
import random
from collections import defaultdict

import nltk
import numpy as np
import numpy.random as rand
from scipy.special import expit
from tqdm import tqdm
import time


class Word2Vec(object):
    def __init__(self, dimension=50, left_window_size=2, right_window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True,
                 output="vectors.txt"):

        # Size of the context window
        self.all_lernig_rate_for_epoch = None
        self.left_window_size = left_window_size
        self.right_window_size = right_window_size

        # Mapping from words to IDs.
        self.word2id = defaultdict(int)

        # Mapping from IDs to words.
        self.id2word = []

        # Mapping from word IDs to (focus) word vectors. (called w_vector
        # to be consistent with the notation in the lecture).
        self.w_vector = []

        # Mapping from word IDs to (context) word vectors (called w_tilde_vector
        # to be consistent with the notation in the lecture)
        self.w_tilde_vector = []

        # Total number of tokens processed
        self.tokens_processed = 0

        # Number of occurrences of each unique word
        self.freq = defaultdict(int)

        # Dimension of word vectors.
        self.dimension = dimension

        # Initial learning rate.
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate

        # All datapoints retrieved from the text. A datapoint is a pair (f,c)
        # where f is the ID of the focus word, and c is a list of the IDs of
        # the context words.
        self.datapoints = []

        # Number of negative samples in each iteration
        self.no_of_neg_samples = nsample

        # Number of epochs through all datapoints
        self.epochs = epochs
        self.current_epoch = 0

        # Use (or don't use) the corrected sampling distribution for negative examples
        self.use_corrected = use_corrected

        # Use (or don't use) learning rate scheduling
        self.use_lr_scheduling = use_lr_scheduling

        # Padding at the beginning and end of the token stream
        self.pad_word = '<pad>'

        # Name of the file to store the word vectors
        self.outputfile = output

        # Temporary file used for storing the model
        self.temp_file = "temp__.txt"

    # ------------------------------------------------------------
    #
    #  Methods for processing all files and producing the list of datapoints
    #

    def get_word_id(self, word):
        """
        Returns the word ID for a given word. If the word has not
        been encountered before, the necessary data structures for
        that word are initialized.
        """
        word = word.lower()
        if word in self.word2id:
            return self.word2id[word]

        else:
            # This word has never been encountered before. Init all necessary
            # data structures.
            latest_new_word = len(self.id2word)
            self.id2word.append(word)
            self.word2id[word] = latest_new_word

            # Initialize arrays with random numbers in [-0.5,0.5].
            w = rand.rand(self.dimension) - 0.5
            self.w_vector.append(w)
            w_tilde = rand.rand(self.dimension) - 0.5
            self.w_tilde_vector.append(w_tilde)
            return latest_new_word

    def get_context(self, i):
        """
        Returns the context of token no i as a list of word indices.

        :param      i:     Index of the focus word in the list of tokens
        :type       i:     int
        """
        s = time.time()
        start = max(0, i - self.left_window_size)
        end = min(len(self.tokens), i + self.right_window_size + 1)  # +1 because it is excluded by the range function
        context = [self.get_word_id(self.tokens[j]) for j in range(start, end) if j != i]
        e = time.time()
        print(f"Fetching context time: {round(e-s, 5)} seconds")
        return context

    def process_files(self, file_or_dir):
        """
        This function recursively processes all files in a directory.

        Each file is tokenized and the tokens are put in the list
        self.tokens.
        """
        if os.path.isdir(file_or_dir):
            for root, dirs, files in os.walk(file_or_dir):
                for file in files:
                    self.process_files(os.path.join(root, file))
        else:
            print(file_or_dir)
            stream = open(file_or_dir, mode='r', encoding='utf-8', errors='ignore')
            text = stream.read()
            try:
                self.tokens = nltk.word_tokenize(text)
            except LookupError:
                nltk.download('punkt')
                self.tokens = nltk.word_tokenize(text)
            for i, token in enumerate(self.tokens):
                self.tokens_processed += 1
                focus_id = self.get_word_id(token)
                context = self.get_context(i)
                self.datapoints.append((focus_id, context))
                self.freq[focus_id] += 1
                if self.tokens_processed % 10000 == 0:
                    print('Processed', "{:,}".format(self.tokens_processed), 'tokens')

    #
    #  End of methods for processing all files and producing the list of datapoints
    #
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    #
    #   Training
    #

    def compute_sampling_distributions(self):

        total_count = sum(self.freq.values())
        P_u = np.array(list(self.freq.values())) / total_count

        # Modified unigram distribution (frequency raised to 3/4 power)
        total_modified_count = np.sum(np.power(list(self.freq.values()), 0.75))
        P_w = (np.array(list(self.freq.values())) ** 0.75) / total_modified_count

        # P_u is the unigram distribution,
        # P_w is the modified unigram distribution
        return P_u, P_w

    def sigmoid(self, x):
        return expit(x) # expit is a numerically stable version of the sigmoid function

    def negative_sampling(self, number, focus, pos, distribution):
        """
        Samples `number` of negatives examples. The words `focus` and `pos`
        are taboo, i.e., those should not be selected.

        :param      number:     The number of negative examples to be sampled
        :param      focus:      The ID of the current focus word
        :param      pos:        The ID of the current positive example
        :param      distribution A list of sampling probabilities (one for each unique word)
        """
        negatives = np.random.choice(len(distribution), size=number * 2,
                                     p=distribution,  replace=False )  # we take twice as many samples so the probability of getting to few sampel are 0

        negatives = negatives[(negatives != focus) & (negatives != pos)][:number]
        return negatives

    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        print(f"Dataset contains {len(self.datapoints)} datapoints")

        P_u, P_w = self.compute_sampling_distributions()
        distribution = P_w if self.use_corrected else P_u

        while self.current_epoch < self.epochs:
            random.shuffle(self.datapoints)

            for i in tqdm(range(len(self.datapoints))):
                focus_id, context_ids = self.datapoints[i]
                focus_vec = self.w_vector[focus_id]

                focus_grads_pos = []
                focus_grads_neg = []

                for pos_id in context_ids:
                    w_tilde = self.w_tilde_vector[pos_id]

                    # Positive example
                    pos_grad_focus, pos_grad_ctx = self.posetive_update(focus_vec, w_tilde)
                    focus_grads_pos.append(pos_grad_focus)
                    self.w_tilde_vector[pos_id] -= self.learning_rate * pos_grad_ctx

                    # Negative sampling
                    negatives = self.negative_sampling(self.no_of_neg_samples, focus_id, pos_id, distribution)
                    for neg_id in negatives:
                        w_tilde_neg = self.w_tilde_vector[neg_id]
                        neg_grad_focus, neg_grad_ctx = self.negative_update(focus_vec, w_tilde_neg)
                        focus_grads_neg.append(neg_grad_focus)
                        self.w_tilde_vector[neg_id] -= self.learning_rate * neg_grad_ctx

                # Update focus vector
                self.update_focuse_vector(focus_grads_pos, focus_grads_neg, focus_id)

            if self.use_lr_scheduling:
                print("[lr={}] Epoch {} is finished!".format(self.learning_rate, self.current_epoch + 1))
            else:
                print("Epoch {} is finished!".format(self.current_epoch + 1))

            assert self.w_vector is not None and not np.isnan(
                self.w_vector).any(), "Word vectors should not be None or contain NaN"
            assert self.w_tilde_vector is not None and not np.isnan(
                self.w_tilde_vector).any(), "Context vectors should not be None or contain NaN"

            self.current_epoch += 1
            self.write_word_vectors_to_file(self.outputfile)
            self.write_temp_file(self.temp_file)

    def posetive_update(self, focus_vec, w_tilde):
        """
        Compute gradient for a positive (focus, context) pair.
        Returns: (grad for focus vector, grad for context vector)
        """
        dot = np.dot(w_tilde, focus_vec)
        sig = self.sigmoid(dot)
        error = sig - 1

        grad_focus = w_tilde * error
        grad_ctx = focus_vec * error
        return grad_focus, grad_ctx

    def negative_update(self, focus_vec, w_tilde_neg):
        """
        Compute gradient for a negative (focus, context) pair.
        Returns: (grad for focus vector, grad for context vector)
        """
        dot = np.dot(w_tilde_neg, focus_vec)
        sig = self.sigmoid(dot)
        error = sig

        grad_focus = w_tilde_neg * error
        grad_ctx = focus_vec * error
        return grad_focus, grad_ctx

    def update_focuse_vector(self, focus_grads_pos, focus_grads_neg, focus_id):
        """
        Update the focus vector using the gradients from the positive and negative examples
        """
        sum_grads = np.sum(focus_grads_pos, axis=0) + np.sum(focus_grads_neg, axis=0)
        self.w_vector[focus_id] -= self.learning_rate * sum_grads


    #
    #  End of training
    #
    # -------------------------------------------------------

    # -------------------------------------------------------
    #
    #  I/O
    #

    def write_word_vectors_to_file(self, filename):
        """
        Writes the vectors to file. These are the vectors you would
        export and use in another application.
        """
        with open(filename, 'w') as f:
            for idx in range(len(self.id2word)):
                f.write('{} '.format(self.id2word[idx]))
                for i in self.w_vector[idx]:
                    f.write('{} '.format(i))
                f.write('\n')
        f.close()

    def write_temp_file(self, filename):
        """
        Saves the state of the computation to file, so that
        training can be resumed later.
        """
        with open(filename, 'w') as f:
            data = {"initial_learning_rate": self.initial_learning_rate,
                    "learning_rate": self.learning_rate,
                    "dimension": self.dimension,
                    "no_of_neg_samples": self.no_of_neg_samples,
                    "epochs": self.epochs,
                    "current_epoch": self.current_epoch,
                    "use_corrected": self.use_corrected,
                    "use_lr_scheduling": self.use_lr_scheduling,
                    "left_window_size": self.left_window_size,
                    "right_window_size": self.right_window_size}
            json.dump(data, f)
            f.write('\n')
            for idx in range(len(self.id2word)):
                f.write('{} '.format(self.id2word[idx]))
                for i in list(self.w_vector[idx]):
                    f.write('{} '.format(i))
                for i in list(self.w_tilde_vector[idx]):
                    f.write('{} '.format(i))
                f.write('\n')
        f.close()

    def read_temp_file(self, fname):
        """
        Reads the partially trained model from file, so
        that training can be resumed.
        """
        try:
            with open(fname) as f:
                data = json.loads(f.readline().strip())
                self.initial_learning_rate = float(data.get("initial_learning_rate", 0.05))
                self.learning_rate = float(data.get("learning_rate", 0.05))
                self.dimension = int(data.get("dimension", 50))
                self.no_of_neg_samples = int(data.get("no_of_neg_sample", 10))
                self.epochs = int(data.get("epochs", 5))
                self.current_epoch = int(data.get("current_epoch", 0))
                self.use_corrected = bool(data.get("use_corrected", True))
                self.use_lr_scheduling = bool(data.get("use_lr_scheduling", True))
                self.left_window_size = int(data.get("left_window_size", 2))
                self.right_window_size = int(data.get("right_window_size", 2))
                for line in f:
                    data = line.split()
                    w = data[0]
                    vec = np.array([float(x) for x in data[1:self.dimension + 1]])
                    self.word2id[w] = len(self.id2word)
                    self.id2word.append(w)
                    self.w_vector.append(vec)
                    vec = np.array([float(x) for x in data[self.dimension + 1:]])
                    self.w_tilde_vector.append(vec)
            f.close()
            print("Read temp file, resuming training.")
        except FileNotFoundError:
            print("Could not find temp file, starting training from scratch.")
        except ValueError:
            print("Error reading temp file, starting training from scratch.")

    #
    #  End of I/O
    #
    # -------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('--file', '-f', type=str,
                        default='../RandomIndexing/data',
                        help='The files used in the training.')
    parser.add_argument('--left_window_size', '-lws', type=int, default='2', help='Left context window size')
    parser.add_argument('--right_window_size', '-rws', type=int, default='2', help='Right context window size')
    parser.add_argument('--continue_training', '-c', action='store_true', default=False,
                        help='Continues training from where it was left off.')
    parser.add_argument('--output', '-o', type=str, default='vectors.txt',
                        help='The file where the vectors are stored.')
    parser.add_argument('--dimension', '-d', type=int, default=50, help='Dimensionality of word vectors')
    parser.add_argument('--negative_samples', '-neg', type=int, default=10, help='Number of negative samples')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.025, help='Initial learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs')
    parser.add_argument('--use-corrected', '-uc', action='store_true', default=True,
                        help="Use the corrected unigram distribution for negative sampling")
    parser.add_argument('--use-learning-rate-scheduling', '-ulrs', action='store_true', default=True,
                        help="Use learning rate scheduling")
    args = parser.parse_args()

    w2v = Word2Vec(
        dimension=args.dimension, left_window_size=args.left_window_size,
        right_window_size=args.right_window_size, output=args.output,
        nsample=args.negative_samples, learning_rate=args.learning_rate, epochs=args.epochs,
        use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
    )
    if args.continue_training:
        w2v.read_temp_file(w2v.temp_file)
    w2v.process_files(args.file)
    print('Processed', "{:,}".format(w2v.tokens_processed), 'tokens')
    print('Found', len(w2v.word2id), 'unique words')
    w2v.train()