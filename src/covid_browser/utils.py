""" Provides utility methods used in scripts."""

import csv


# for reading sentences. Returns list of words in a sentence
def read_from_file(filename, delimiter='\t'):
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        data = list(reader)
        return data
    return None
