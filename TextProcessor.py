from DataProcessing import *
import numpy as np
import re


import re

class TextProcessor:
    def __init__(self, data_str):
        self.data_str = data_str
        self.data_lower = data_str.lower()
        self.unique_chars = set(data_str)  # Collect unique characters in the input data
        self.word_dict = self._create_word_dict()

    def _create_word_dict(self):
        words = re.split(r"[,\$\?'\n&;:\s\-\.\!]+", self.data_lower)
        unique_words = set(words)
        return {word: 1 for word in unique_words}

    def correctly_spelt_count(self, text):
        text_lower = text.lower()
        words = re.findall(r"[,\$\?'\n&;:\s\-\.\!]+", text_lower)

        counter = 0
        for word in words:
            if word in self.word_dict:
                counter += 1
        return counter

"""
# Example usage
data_str = "This is an example sentence. Some words here are correct, others may not be."
processor = TextProcessor(data_str)

# Test the methods
print("Correctly spelt count:", processor.correctly_spelt_count("aaaaaaaaaaaaaaaaa blaze. linger, ear fur"))
"""