"""
convert model to gensim format, because installing magnitude disabled CUDA inside a container
"""
from gensim.models import KeyedVectors
import numpy as np
import os
import sys


def convert(dir_path):
    with open(os.path.join(dir_path, 'cord19-300d.txt'), mode="r") as txtfile:
        lines = txtfile.readlines()
        num_entries = len(lines)
        vecs = np.zeros((num_entries-1, 300), float)
        words = []
        print(type(words))
        idx = 0
        for line in lines[1:]:
            tok = line.split(" ")
            words.append(tok[0])
            vals = tok[1:]
            vecs_ = np.array([float(item) for item in vals])
            vecs[idx, :] = vecs_
            idx = idx + 1

        model = KeyedVectors(vecs.shape[1])
        model.add(words, vecs)
        model.save(os.path.join(dir_path, 'cord19-300d.wv'))

        # load the model back and verify results
        model_ = model.load(os.path.join(dir_path, 'cord19-300d.wv'))

if __name__ == "__main__":
    path = '/home/ankur/dev/apps/ML/covid-papers-analysis'
    convert(sys.argv[1] if len(sys.argv) > 1 else path)
