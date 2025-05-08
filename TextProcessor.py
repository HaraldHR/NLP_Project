from DataProcessing import *
import numpy as np
import re

class TextProcessor:
    def __init__(self, data_str):
        self.data_str = data_str
        self.data_lower = data_str.lower()
        self.word_set = self._create_word_set()
        self.gram_2, self.gram_3 = self._create_n_gram_char_sets()

    def _create_word_set(self):
        """Creates a set of all unique words."""
        words = re.split(r"[,\$\?'\n&;:\s\-\.\!]+", self.data_lower)
        return set(filter(None, words))  # Remove empty strings

    def _create_n_gram_char_sets(self):
        """Creates sets of 2-grams and 3-grams of characters."""
        gram_2 = set()
        gram_3 = set()
        text = self.data_lower

        for i in range(len(text) - 1):
            gram_2.add(text[i:i+2])
        for i in range(len(text) - 2):
            gram_3.add(text[i:i+3])

        return gram_2, gram_3
    
    def measure_n_grams(self, text):
        correct_2_gram = 0
        tot_2_gram = 0
        correct_3_gram = 0
        tot_3_gram = 0

        for i in range(len(text) - 1):
            tot_2_gram += 1
            if text[i:i+2] in self.gram_2: 
                correct_2_gram += 1
        for i in range(len(text) - 2):
            tot_3_gram += 1
            if text[i:i+3] in self.gram_3: 
                correct_3_gram += 1

        return correct_2_gram / tot_2_gram, correct_3_gram / tot_3_gram
    def correctly_spelt_count(self, text):
        """Returns the fraction of correctly spelled words in the input."""
        text_lower = text.lower()
        words = re.split(r"[,\$\?'\n&;:\s\-\.\!]+", text_lower)
        words = list(filter(None, words))  # Remove empty strings

        if not words: # if no words in the string
            return 0

        correct = sum(1 for word in words if word in self.word_set)
        return correct / len(words)

"""
# Example usage
data_str, _ = ReadData()
processor = TextProcessor(data_str)

# Test
print("Correctly spelt count:", processor.measure_n_grams("aaaaaaaaa blaze. linger, ear for"))
"""