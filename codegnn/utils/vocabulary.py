import pickle
from argparse import ArgumentParser
from collections import Counter
from json import JSONDecodeError, loads
from os import path
from os.path import exists
from typing import Dict
from typing import Counter as CounterType
import matplotlib.pyplot as plt
import numpy as np

from omegaconf import DictConfig
from tqdm import tqdm

from utils.common import (
    PAD,
    UNK,
    NODE,
    SOS,
    EOS,
    TOKEN,
    LABEL,
    SEPARATOR,
    AST,
    SOURCE,
    get_lines_in_file,
    SPLIT_FIELDS,
)


class Vocabulary:
    vocab_file = "vocabulary.pkl"
    _log_file = "bad_samples.log"

    def __init__(self, config: DictConfig):
        vocabulary_file = path.join(config.data_folder, config.dataset, self.vocab_file)
        if not exists(vocabulary_file):
            raise ValueError(f"Can't find vocabulary file ({vocabulary_file})")
        with open(vocabulary_file, "rb") as f_in:
            counters = pickle.load(f_in)
        max_size = {LABEL: config.max_labels, TOKEN: config.max_tokens}
        self._vocabs = {}
        for key, counter in counters.items():
            if key in SPLIT_FIELDS:
                service_tokens = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
            else:
                service_tokens = {PAD: 0, UNK: 1}
            skip_id = len(service_tokens)
            self._vocabs[key] = service_tokens
            self._vocabs[key].update(
                (token[0], i + skip_id) for i, token in enumerate(counter.most_common(max_size.get(key, None)))
            )

    @property
    def vocabs(self) -> Dict[str, Dict]:
        return self._vocabs

    @property
    def node_to_id(self) -> Dict[str, int]:
        return self._vocabs[NODE]

    @property
    def token_to_id(self) -> Dict[str, int]:
        return self._vocabs[TOKEN]

    @property
    def label_to_id(self) -> Dict[str, int]:
        return self._vocabs[LABEL]

    @property
    def id_to_node(self) -> Dict[int, str]:
        return {v: k for k, v in self._vocabs[NODE].items()}

    @property
    def id_to_token(self) -> Dict[int, str]:
        return {v: k for k, v in self._vocabs[TOKEN].items()}

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {v: k for k, v in self._vocabs[LABEL].items()}

    @staticmethod
    def build_from_scratch(train_data: str):
        total_samples = get_lines_in_file(train_data)
        label_counter: CounterType[str] = Counter()
        token_counter: CounterType[str] = Counter()
        node_counter: CounterType[str] = Counter()
        feature_counters: Dict[str, CounterType[str]] = {}
        label_lens = []
        source_lens = []
        n_nodes = []
        with open(train_data, "r") as f_in:
            for sample_id, sample_json in tqdm(enumerate(f_in), total=total_samples):
                try:
                    sample = loads(sample_json)
                except JSONDecodeError:
                    with open(Vocabulary._log_file, "a") as log_file:
                        log_file.write(sample_json + "\n")
                    continue

                label_tokens = sample[LABEL].split(SEPARATOR)
                label_lens.append(len(label_tokens))
                label_counter.update(label_tokens)

                source_tokens = sample[SOURCE].split(SEPARATOR)
                source_lens.append(len(source_tokens))
                token_counter.update(source_tokens)

                for node in sample[AST]:
                    node_counter.update([node[NODE]])
                n_nodes.append(len(sample[AST]))

        feature_counters[LABEL] = label_counter
        feature_counters[TOKEN] = token_counter
        feature_counters[NODE] = node_counter
        for feature, counter in feature_counters.items():
            print(f"Count {len(counter)} {feature}, top-5: {counter.most_common(5)}")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
        ax1.set_title('label lengths distribution')
        ax1.hist(label_lens, bins=50)
        ax1.axvline(x=np.mean(label_lens), ls='--', color='red')
        ax2.set_title('source code length distribution')
        ax2.hist(source_lens, bins=50)
        ax2.axvline(x=np.mean(source_lens), ls='--', color='red')
        ax3.set_title('nodes number distribution')
        ax3.hist(n_nodes, bins=50)
        ax3.axvline(x=np.mean(n_nodes), ls='--', color='red')

        plt.show()

        print(f"min label length: {min(label_lens)}, max label length: {max(label_lens)}, average: {np.mean(label_lens)}")
        print(f"min source code length: {min(source_lens)}, max source code length: {max(source_lens)}, average: {np.mean(source_lens)}")
        print(f"min nodes number: {min(n_nodes)}, max nodes number: {max(n_nodes)}, average: {np.average(n_nodes)}")

        dataset_dir = path.dirname(train_data)
        vocabulary_file = path.join(dataset_dir, Vocabulary.vocab_file)
        with open(vocabulary_file, "wb") as f_out:
            pickle.dump(feature_counters, f_out)

        return label_lens, source_lens, n_nodes


if __name__ == "__main__":
    arg_parse = ArgumentParser()
    arg_parse.add_argument("data", type=str, help="Path to file with data")
    args = arg_parse.parse_args()
    Vocabulary.build_from_scratch(args.data)