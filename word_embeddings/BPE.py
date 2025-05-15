'''
File contains structure for performing a Byte-Pair encoding to tokenize the text.
'''

from transformers import AutoTokenizer
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained("gpt2")

class BPE:

    def __init__(self, corpus, max_vocab_size=200):
        self.max_vocab_size = max_vocab_size
        self.word_freqs = defaultdict(int)
        self.vocab = []

        self.pre_tokenize(corpus)
        self.create_vocab()



    def pre_tokenize(self, corpus):
        for text in corpus:
            words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freqs[word] += 1
        print(self.word_freqs)



    def create_vocab(self):
        for word in self.word_freqs.keys():
            for char in word:
                if char not in self.vocab:
                    self.vocab.append(char)
        self.vocab.sort()
        print(self.vocab)
        print(["<|endoftext|>"] + self.vocab.copy())


corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
bpe = BPE(corpus)



