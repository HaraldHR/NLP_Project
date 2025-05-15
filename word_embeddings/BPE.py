'''
File contains structure for performing a Byte-Pair encoding to tokenize the text.
Largly inspired from the Hugging Face tutorial code, with a class structure.

Possible improvements:
- Support for unknown characters.
- Create a sampling for pairs with equal frequency instead of picking the first one that appears.
'''

from transformers import AutoTokenizer
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained("gpt2")

class BPE:
    def __init__(self, max_vocab_size=50):
        self.max_vocab_size = max_vocab_size
        self.word_freqs = defaultdict(int)
        self.pair_freqs = defaultdict(int)
        self.vocab = []
        self.tokens = {} # will contain a dictionairy with each word as key, and that words tokenization as a list of tokens as the value.


    def tokenize(self, text, create_new_vocab=True):
        '''
        Tokenizes the inputted text into the models specified vocab length.
        :param text:
        :param create_new_vocab:
        :return:
        '''
        if create_new_vocab:
            self.reset_model()
        self.pre_tokenize(text)
        self.create_vocab()
        self.generate_tokens(corpus)

        return self.vocab, self.tokens


    def reset_model(self):
        self.word_freqs = defaultdict(int)
        self.pair_freqs = defaultdict(int)
        self.vocab = []
        self.tokens= {}


    def set_vocab_length(self, vocab_length):
        '''
        Sets a new vocabulary length of the instantiated model.
        :param vocab_length:
        :return:
        '''
        assert type(vocab_length) == int, f"New vocabulary length must be of type INT."
        self.max_vocab_size = vocab_length


    # Function imported from Hugging Face tutorial.
    def pre_tokenize(self, corpus):
        for text in corpus:
            words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1



    def create_vocab(self):
        for word in self.word_freqs.keys():
            for char in word:
                if char not in self.vocab:
                    self.vocab.append(char)
        self.vocab.sort()
        self.vocab = ["<|endoftext|>"] + self.vocab.copy()



    def compute_pair_freqs(self, splits):
        '''
        This method calculates the frequency of pairs of characters in the corpus.
        Only accounts for charactes in the same words, and NOT between words and sentences.
        param: dictionairy
            contains the words as keys, and a list of all characters in that word as values.
        :return:
        '''
        self.pair_freqs = defaultdict(int) # empties the dictionairy.
        for word, freq in self.word_freqs.items():
            split = splits[word] # gathers the characters of the word.
            if len(split) == 1: # if word only contains one character.
                continue
            for i in range(len(split)-1):
                pair = (split[i], split[i+1])
                # freqeuncy of the word will also denote how many occurances of the pair there are in the corpus.
                self.pair_freqs[pair] += freq


    # Function mostly copied from Hugging Face tutorial.
    def merge_pair(self, a, b, splits):
        for word in self.word_freqs:
            split = splits[word] # split is a list of characters in the word.
            if len(split) == 1: # If word is only one character, we skip this word.
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b: # We find our match, then merge it.
                    split = split[:i] + [a + b] + split[i + 2:]
                    self.vocab.append(a+b)
                else:
                    i += 1
            splits[word] = split # assigns the new list of tokens to the word.
        return splits


    def generate_tokens(self, corpus):
        print(f"Initial vocab length: {len(self.vocab)}")
        assert self.max_vocab_size > len(self.vocab), f"Maximum vocab size must be larger than initialized vocab size"
        splits = {word: [c for c in word] for word in self.word_freqs.keys()} # initializes splits.
        for i in range(self.max_vocab_size - len(self.vocab)): # loops until vocab size is full.
            self.compute_pair_freqs(splits)
            max_freq = 0
            max_pair = None
            for pair in self.pair_freqs.keys():
                freq = self.pair_freqs[pair]
                if freq > max_freq:
                    max_pair = pair
                    max_freq = freq
            if max_pair is None:
                continue
            splits = self.merge_pair(max_pair[0], max_pair[1], splits) # Updates splits with new merged token.
        self.tokens = set(token for tokens in splits.values() for token in tokens)



corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
bpe = BPE()
vocab, tokens = bpe.tokenize(corpus)
print(f"New tokens: {tokens}")
bpe.set_vocab_length(20)
vocab, tokens = bpe.tokenize(["This is not a token."])
print(tokens)



