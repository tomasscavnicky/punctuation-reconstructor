# autor: Karel Benes

import numpy as np

class Word2Vec:
    def __init__(self, filename):
        self._w2v = {}
        with open(filename) as f:
            for line in f:
                fields = line.split(" ")
                self._w2v[fields[0]] = np.asarray(map(float, fields[1:]), dtype=np.float32)
        self._default_vec = np.mean(self._w2v.values(), axis=0)
        self.reset_experience()

    def default_vec(self):
        return self._default_vec

    def vector_len(self):
        return self._w2v.values()[0].shape[0]

    def experience(self):
        return "Known " + str(self._nb_known) + " words, unknown " + str(self._nb_unknown)

    def reset_experience(self):
        self._nb_known = 0
        self._nb_unknown = 0

    def w2v(self, word):
        if word in self._w2v:
            self._nb_known += 1
            return self._w2v[word]
        else:
            self._nb_unknown += 1
            return self._default_vec
