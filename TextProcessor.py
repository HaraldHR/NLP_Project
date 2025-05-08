from DataProcessing import *
import numpy as np
import re


import re

class TextProcessor:
    def __init__(self, data_str):
        self.data_str = data_str
        self.data_lower = data_str.lower()
        self.word_dict = self._create_word_dict()

    def _create_word_dict(self):
        """Creates a dictionary with all unique words as keys"""
        words = re.split(r"[,\$\?'\n&;:\s\-\.\!]+", self.data_lower)
        #print(self.data_lower)
        unique_words = set(words)
        return {word: 1 for word in unique_words}

    def correctly_spelt_count(self, text):
        """Returns the fraction of correctly spelt words."""
        text_lower = text.lower()
        words = re.split(r"[,\$\?'\n&;:\s\-\.\!]+", text_lower)
        counter = 0
        for word in words:
            if word in self.word_dict:
                counter += 1
        if counter == 0: return 0

        return counter / len(words)


"""
# Example usage
data_str, _ = ReadData()
processor = TextProcessor(data_str)

# Test the methods
print("Correctly spelt count:", processor.correctly_spelt_count("aaaaaaaaaaaaaaaaa blaze. linger, ear for"))
"""