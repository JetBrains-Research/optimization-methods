import pickle
import nltk
from collections import Counter
from tqdm import tqdm
import numpy as np


UNK_INDEX = 0
PAD_INDEX = 1
SOS_INDEX = 2
EOS_INDEX = 3

TOP_DENSITY = 0.75


class WordTokenizerResponse:
    def __init__(self, ids):
        self.ids = ids


class WordTokenizer:
    def __init__(self, filename: str, pretrained: bool = True, vocab_size: int = None, special_tokens: list = [], only_top: bool = True):
        if pretrained:
            with open(filename, "rb") as config:
                self.dict, self.special_tokens = pickle.load(config)
            self.vocab_size = len(self.dict)
        else:
            self.train(filename, vocab_size, special_tokens, only_top)

        print("vocab size", self.vocab_size)
        self.dict_back = {v: k for k, v in self.dict.items()}
        self.max_length = None

    def encode(self, sentence: str):
        ids = [SOS_INDEX] + list(map(
            lambda word: self.dict[word] if word in self.dict else UNK_INDEX,
            nltk.word_tokenize(sentence)
        )) + [EOS_INDEX]
        
        if self.max_length is not None:
            ids = ids[:self.max_length]

            if ids[min(self.max_length, len(ids)) - 1] not in [PAD_INDEX, EOS_INDEX]:
                ids[min(self.max_length, len(ids)) - 1] = EOS_INDEX
            
            if len(ids) > self.max_length:
                ids += [PAD_INDEX] * (len(ids) - self.max_length)
        
        return WordTokenizerResponse(ids)

    def encode_batch(self, sentences: list):
        result = []
        for sentence in sentences:
            result.append(self.encode(sentence))
        return result

    def decode(self, ids: list):
        return " ".join(list(filter(
            lambda word: word != "xxx",
            map(
                lambda word_id: 
                    self.dict_back[word_id] 
                    if word_id not in [UNK_INDEX, PAD_INDEX, SOS_INDEX, EOS_INDEX]
                    else "xxx", 
                ids
            )
        )))

    def train(self, text_filename: str, vocab_size: int = None, special_tokens: list = [], only_top: bool = True):
        self.special_tokens = special_tokens
        print("Training...")
        if vocab_size is None and not only_top:
            tokens = set()
            print("Reading textfile:")
            num_lines = sum(1 for line in open(text_filename))
            with open(text_filename) as f:
                for line in tqdm(f, total=num_lines):
                    tokens.update(set(nltk.word_tokenize(line)))
            print("Processing dict...")
            self.dict = {word: number for number, word in enumerate(special_tokens + list(sorted(tokens)))}
            self.vocab_size = len(self.dict)
        else:
            tokens = []
            print("Reading textfile:")
            num_lines = sum(1 for line in open(text_filename))
            with open(text_filename) as f:
                for line in tqdm(f, total=num_lines):
                    tokens += nltk.word_tokenize(line)
            print("Processing dict...")
            counter = Counter(tokens)

            if not only_top:
                self.vocab_size = vocab_size
                self.dict = {word_freq[0]: number for number, word_freq in enumerate(list(zip(special_tokens, [0, 0, 0, 0])) + list(counter.most_common(vocab_size)))}
            else:
                counts = list(sorted(list(map(lambda word_freq: word_freq[1], counter.items())), reverse=True))
                threshold = counts[np.searchsorted(np.cumsum(counts), TOP_DENSITY * counts[-1])]
                self.dict = {word_freq[0]: number for number, word_freq in enumerate() if word_freq[1] >= threshold}
                self.vocab_size = len(self.dict)

        print("Ready!")

    def enable_truncation(self, max_length: int):
        self.max_length = max_length

    def get_vocab_size(self):
        return self.vocab_size

    def save(self, filename: str):
        with open(filename, "wb") as config:
            pickle.dump((self.dict, self.special_tokens), config)
