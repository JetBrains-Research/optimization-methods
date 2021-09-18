import pickle
import nltk
from collections import Counter
from tqdm import tqdm


class WordTokenizerResponse:
    def __init__(self, ids):
        self.ids = ids


class WordTokenizer:
    def __init__(self, filename: str, pretrained: bool = True, vocab_size: int = None):
        if pretrained:
            with open(filename, "rb") as config:
                self.dict = pickle.load(config)
            self.vocab_size = len(self.dict)
        else:
            self.train(filename, vocab_size)

        print("vocab size", self.vocab_size)
        self.dict_back = {v: k for k, v in self.dict.items()}
        self.max_length = None

    def encode(self, sentence: str):
        ids = list(map(
            lambda word: self.dict[word],
            nltk.word_tokenize(sentence)
        ))
        
        if self.max_length is not None:
            ids = ids[:self.max_length]
        
        return WordTokenizerResponse(ids)

    def encode_batch(self, sentences: list):
        result = []
        for sentence in sentences:
            result.append(self.encode(sentence))
        return result

    def decode(self, ids: list):
        return " ".join(list(map(lambda word_id: self.dict_back[word_id], ids)))

    def train(self, text_filename: str, vocab_size: int = None):
        print("Training...")
        if vocab_size is None:
            tokens = set()
            print("Reading textfile:")
            num_lines = sum(1 for line in open(text_filename))
            with open(text_filename) as f:
                for line in tqdm(f, total=num_lines):
                    tokens.update(set(nltk.word_tokenize(line)))
            print("Processing dict...")
            self.dict = {word: number for number, word in enumerate(sorted(tokens))}
            self.vocab_size = len(self.dict)
        else:
            self.vocab_size = vocab_size
            tokens = []
            print("Reading textfile:")
            num_lines = sum(1 for line in open(text_filename))
            with open(text_filename) as f:
                for line in tqdm(f, total=num_lines):
                    tokens += nltk.word_tokenize(line)
            print("Processing dict...")
            counter = Counter(tokens)
            self.dict = {word_freq[0]: number for number, word_freq in enumerate(counter.most_common(vocab_size))}
        print("Ready!")

    def enable_truncation(self, max_length: int):
        self.max_length = max_length

    def get_vocab_size(self):
        return self.vocab_size

    def save(self, filename: str):
        with open(filename, "wb") as config:
            pickle.dump(self.dict, filename)
