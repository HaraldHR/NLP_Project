'''
Uses logistic regression to compute word2vec word embedding from a data-set.
Code inspired by DD2417 Language Engineering, Assignment 3.
'''

import os
import time
import nltk
import numpy as np
from collections import defaultdict
import torch
from tqdm import tqdm
import copy
from BPE import BPE


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


class Word2Vec(object):
    def __init__(self, dimension=100, lws_size=2, rws_size=2, n_neg_samples=10,
                 lr=0.025, epochs=5, lr_scheduling=True, tokenization=None, output="vectors.txt"):

        # Size of the context window.
        self.lws_size = lws_size
        self.rws_size = rws_size

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

        # Number of occurrences of each unique word.
        self.freq = defaultdict(int)

        # Modified unigram distribution.
        self.uni_dist = None

        # Dimension of word vectors.
        self.dimension = dimension

        # Initial learning rate.
        self.init_lr = lr
        self.lr = lr

        # All datapoints retrieved from the text. A datapoint is a pair (f,c),
        # where f is the ID of the focus word, and c is a numpy vector of the IDs of the context words.
        self.datapoints = []

        # Number of negative samples in each iteration.
        self.n_neg_samples = n_neg_samples

        # Number of epochs used for training the model.
        self.epochs = epochs
        self.current_epoch = 0

        # If we want to use scheduled learning rate or not.
        self.lr_scheduling = lr_scheduling

        # Name of the file to store the word vectors.
        self.outputfile = output

        # Temporary file used for storing the model.
        self.temp_file = "temp__.txt"

        # Initializes the processed token to be None.
        self.tokens = None

        # If None use standard tokenization. Otherwise input a tokenization module (like BPE).
        self.tokenization = tokenization


    # Function written by me.
    def get_word_id(self, word):
        '''
        Returns the word ID for a given word.
        If the word has not been encountered before,
            the necessary data structures for that word are initialized.
        '''
        word = word.lower()
        if word in self.word2id:
            return self.word2id[word]
        else:
            # This word has never been encountered before. Init all necessary data structures.
            latest_new_word = len(self.id2word)
            self.id2word.append(word)
            self.word2id[word] = latest_new_word

            # Initialize arrays with random numbers in [-0.5,0.5].
            w = np.random.rand(self.dimension) - 0.5
            self.w_vector.append(w)
            w_tilde = np.random.rand(self.dimension) - 0.5
            self.w_tilde_vector.append(w_tilde)
            return latest_new_word


    # Function written by me.
    def get_context(self, i):
        '''
        Returns the context of token no i as a list of word indices.

        :param  i:     Index of the focus word in the list of tokens.
        '''

        #TODO: Convert from list to a numpy array for more efficient computing.

        ctx_word_ids = []

        for j in range(1, self.lws_size + 1):
            try:
                word_id = self.datapoints[i - j][0]  # can fetch previous words with self.datapoints
                ctx_word_ids.append(word_id)
            except:
                print(
                    f"Not enough words to fit the left-window slide. Continuing with {j} out of {self.lws_size} words to the left. ")
        for j in range(1, self.rws_size + 1):
            try:
                # For future words, we need to look into self.tokens.
                word = self.tokens[i + j]
                word_id = self.get_word_id(word)  # initializes a new ID to the word, if it has not been seen.
                ctx_word_ids.append(int(word_id))
            except:
                print(
                    f"Not enough words to fit the right-window slide. {self.rws_size - j} words remaining in document.")

        return np.array(ctx_word_ids)


    # Function from the code skeleton.
    def process_files(self, file_or_dir):
        '''
        This function recursively processes all files in a directory.
        Each file is tokenized and the tokens are put in the list
        self.tokens.
        '''
        if os.path.isdir(file_or_dir):
            for root, dirs, files in os.walk(file_or_dir):
                for file in files:
                    self.process_files(os.path.join(root, file))
        else:
            print(f"Processing file: {file_or_dir}")
            stream = open(file_or_dir, mode='r', encoding='utf-8', errors='ignore')
            text = stream.read()

            if self.tokenization is None:
                try:
                    self.tokens = nltk.word_tokenize(text)
                    print(self.tokens)
                except LookupError:
                    nltk.download('punkt_tab')
                    self.tokens = nltk.word_tokenize(text)
            else:
                try:
                    print("Fetching BPE tokens...")
                    vocab, self.tokens = self.tokenization.tokenize(text)
                    print(self.tokens)
                except:
                    raise ValueError(f"Something went wrong in tokenization...")

            for i, token in enumerate(self.tokens):
                self.tokens_processed += 1
                focus_id = self.get_word_id(token)
                context = self.get_context(i) # should be a numpy array.
                assert isinstance(context, np.ndarray), f"Context IDs are not in an numpy array. They have type {type(context)}"
                self.datapoints.append((focus_id, context))
                self.freq[focus_id] += 1 # updates the frequency for the word in the unigram distribution.
                if self.tokens_processed % 10000 == 0:
                    print(f"Processed {self.tokens_processed} tokens")


    # Function written by me.
    def compute_sampling_distributions(self):
        start = time.time()
        # Mapping function
        def uni_sample(word_id):
            return self.freq[word_id] / len(self.datapoints)

        # P_u is the unigram distribution,
        # P_w is the modified unigram distribution
        P_u = np.array(list(map(uni_sample, self.freq.keys())))
        mod_list = P_u ** 0.75
        sum_mod = np.sum(mod_list)
        P_w = [p / sum_mod for p in mod_list]

        end = time.time()
        print(f"Sampling distribution computed in {round(end - start, 4)} seconds")
        self.uni_dist = P_w

        return P_w


    # Function written by me.
    def negative_sampling(self, number, focus, pos):
        '''
        Computes the negative sampling.

        :param      number:     The number of negative examples to be sampled
        :param      focus:      The ID of the current focus word
        :param      pos:        The ID of the current positive example
        '''
        samples = np.zeros(number) # initializes the numpy array that contains all negative sample IDs.
        samples[0] = pos # just for the while loop to trigger.
        while focus in samples or pos in samples:
            # Samples a word ID from the distribution.
            word_ids = np.array(list(self.freq.keys()))
            samples = np.random.choice(word_ids, number, p=self.uni_dist)
        return samples



    def train_vec(self):
        '''
        Vectorized training of the word2vec skip-gram model.
        '''
        print(f"Dataset contains {len(self.datapoints)} datapoints")

        _ = self.compute_sampling_distributions()

        while self.current_epoch < self.epochs:
            for i in tqdm(range(len(self.datapoints))):
                foc_word_id = self.datapoints[i][0]
                ctx_word_ids = self.datapoints[i][1]  # shape: (C,)

                W = self.w_vector[foc_word_id]  # shape: (D,)
                W = W.reshape(1, -1)  # shape: (1, D)

                # === Positive samples ===
                W_tilde_pos = self.w_tilde_vector[ctx_word_ids]  # shape: (C, D)
                dot_pos = W_tilde_pos @ W.T  # shape: (C, 1)
                sig_pos = sigmoid(dot_pos)  # shape: (C, 1)

                # Gradients for positive context words
                grad_w_pos = np.sum(W_tilde_pos * (sig_pos - 1), axis=0)  # shape: (D,)
                grad_w_tilde_pos = (sig_pos - 1) * W  # shape: (C, D)
                self.w_tilde_vector[ctx_word_ids] -= self.lr * grad_w_tilde_pos

                # === Negative samples ===
                neg_samples = np.array([
                    self.negative_sampling(self.n_neg_samples, foc_word_id, pos_ex_id)
                    for pos_ex_id in ctx_word_ids
                ]).reshape(-1)  # shape: (C * n_neg_samples,)

                W_tilde_neg = self.w_tilde_vector[neg_samples]  # shape: (C * n_neg, D)
                dot_neg = W_tilde_neg @ W.T  # shape: (C * n_neg, 1)
                sig_neg = sigmoid(dot_neg)  # shape: (C * n_neg, 1)

                grad_w_neg = np.sum(W_tilde_neg * sig_neg, axis=0)  # shape: (D,)
                grad_w_tilde_neg = sig_neg * W  # shape: (C * n_neg, D)
                self.w_tilde_vector[neg_samples] -= self.lr * grad_w_tilde_neg

                # === Update focus word vector ===
                grad_focus = grad_w_pos + grad_w_neg  # shape: (D,)
                self.w_vector[foc_word_id] -= self.lr * grad_focus

            if self.lr_scheduling:
                print(f"LR scheduling active: Epoch {self.current_epoch + 1} is finished using learning rate={self.lr}")
            else:
                print(f"Epoch {self.current_epoch + 1} finished. LR scheduling inactive.")

            self.current_epoch += 1
            self.write_word_vectors_to_file(self.outputfile)



    # Function largely written by me
    def train(self):
        '''
        Performs the training of the word2vec skip-gram model
        '''
        print(f"Dataset contains {len(self.datapoints)} datapoints")

        _ = self.compute_sampling_distributions() # Updates self.P_w

        #TODO: Vectorize approach and remove all lists.
        while self.current_epoch < self.epochs:
            for i in tqdm(range(len(self.datapoints))):

                foc_word_id = self.datapoints[i][0]  # the ID of the i'th token / datapoint in the text corpus.
                ctx_word_ids = self.datapoints[i][1] # Of type numpy array.

                W = self.w_vector[foc_word_id]  # the specific focus word vector we need.

                grads_bit_pos = np.zeros( (ctx_word_ids.shape[0], W.shape[0]) ) # shape ctx_words x len of each vector
                grads_bit_neg = np.zeros( (ctx_word_ids.shape[0]*self.n_neg_samples, W.shape[0]) )

                for i, pos_ex_id in enumerate(ctx_word_ids):
                    W_tilde = self.w_tilde_vector[pos_ex_id]  # the specific context vector we need.

                    # Calcs for the focus vector gradient.
                    bit_ctx = W_tilde * (sigmoid(W_tilde.T @ W) - 1)  # calculates pne part of the sum.
                    grads_bit_pos[i] = (bit_ctx)

                    # Calcs gradient and updates the context vector.
                    ctx_pos_grad = W * (sigmoid(W_tilde.T @ W) - 1)
                    self.w_tilde_vector[pos_ex_id] -= self.lr * ctx_pos_grad

                    neg_samples = self.negative_sampling(self.n_neg_samples, foc_word_id, pos_ex_id)
                    for i, neg_id in enumerate(neg_samples):
                        W_tilde = self.w_tilde_vector[neg_id]
                        grads_bit_neg[i] = W_tilde * sigmoid(W_tilde.T @ W)
                        ctx_neg_grad = W * (sigmoid(W_tilde @ W))
                        self.w_tilde_vector[neg_id] -= self.lr * ctx_neg_grad

                # calculates the sum and returns one part of the gradient vector.
                grads_bit_pos = np.sum(grads_bit_pos, axis=0)
                grads_bit_neg = np.sum(grads_bit_neg, axis=0)
                grad_focus = grads_bit_pos + grads_bit_neg

                # Updates the focus vector
                self.w_vector[foc_word_id] -= self.lr * grad_focus

            if self.lr_scheduling:
                print(f"LR scheduling active: Epoch {self.current_epoch + 1} is finished using learning rate={self.lr}")
            else:
                print(f"Epoch {self.current_epoch + 1} finished. LR scheduling inactive.")
            # Write to file after each epoch.
            self.current_epoch += 1
            self.write_word_vectors_to_file(self.outputfile)



    # Function copied from the code skeleton. Maybe re-write?
    def write_word_vectors_to_file(self, filename):
        '''
        Writes the vectors to file. These are the vectors you would
        export and use in another application.
        '''
        with open(filename, 'w') as f:
            for idx in range(len(self.id2word)):
                f.write('{} '.format(self.id2word[idx]))
                for i in self.w_vector[idx]:
                    f.write('{} '.format(i))
                f.write('\n')
        f.close()



if __name__ == '__main__':
    print(nltk.data.find('tokenizers/punkt_tab'))

    file = '../shakespeare.txt'

    # Creates the word2vec model.
    bpe = BPE(max_vocab_size=500)
    w2v = Word2Vec(dimension=100, n_neg_samples=15, epochs=5, lr_scheduling=True, tokenization=bpe)
        
    w2v.process_files(file)
    print(f"Processed {w2v.tokens_processed} tokens")
    print(f"Found {len(w2v.word2id)} unique words")
    w2v.train()




