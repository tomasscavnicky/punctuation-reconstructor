#autor: Karel Benes

class SeqGen(object):
    def __init__(self, sentence, word_to_feas, word_to_targets, seq_len, unk_word, target_ind=True):
        self.sentence = sentence
        self._word_to_feas = word_to_feas
        self.w2t = word_to_targets
        self.index = 0
        self.seq_len = seq_len
        self.unk_word = unk_word
        self.target_ind = target_ind

        self._seq = []
        self._target = []

    def __iter__(self):
        return self

    def w2f(self, x):
        try:
            return self._word_to_feas(x)
        except KeyError:
            return self._word_to_feas(self.unk_word)

    def next(self):
        if self.index + self.seq_len + 1 <= len(self.sentence):
            self._seq = self._seq[1:]
            self._target = self._target[1:]

            while len(self._seq) < self.seq_len:
                if self.index + 1 == len(self.sentence):
                    raise StopIteration()

                if self.w2t(self.sentence[self.index]) != 0:
                    self.index += 1
                elif self.w2t(self.sentence[self.index+1]) != 0:
                    self._seq.append(self.w2f(self.sentence[self.index]))
                    self._target.append(self.w2t(self.sentence[self.index+1]))
                    self.index += 2 # skip the punctuation 'word'
                else:
                    self._seq.append(self.w2f(self.sentence[self.index]))
                    self._target.append(self.w2t(self.sentence[self.index+1]))
                    self.index += 1
                    
            seq = self._seq
            if self.target_ind is True:
                target = self._target
                self._seq = []
                self._target = []
            else:
                target = self._target[self.target_ind]

            return seq, target
        else:
            raise StopIteration()

if __name__ == '__main__':
    text = sys.stdin.read()
    generator = SeqGen(text.split(" "), 
        lambda x:x, lambda x: 1 if x in {'.','?'} else 0, 4, '<unk>', 1)
    for x in generator:
        print x
