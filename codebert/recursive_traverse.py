from os import listdir
import os
from os.path import join, isfile
import shutil
import json
import random
import re
import regex
import pickle
from tqdm import tqdm


def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)


re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

DATA_DIR = "./"


class DataPreprocessor:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.indices = {
            'train': 0,
            'val': 0,
            'test': 0,
        }

    def collect_all_java_files(self, in_dir: str, holdout: str, copy_dir: str):
        for smth in listdir(in_dir):
            cur_path = join(in_dir, smth)
            if isfile(cur_path):
                if smth.split('.')[-1] == 'java':
                    os.makedirs(join(copy_dir, holdout), exist_ok=True)
                    shutil.copyfile(
                        cur_path,
                        join(copy_dir, holdout,
                             f"{self.indices[holdout]}.java")
                    )
                    self.indices[holdout] += 1
            else:
                self.collect_all_java_files(cur_path, holdout, copy_dir)

    def preprocess(self):
        for holdout in self.indices:
            if holdout == 'train':
                partition_dir = 'training'
            elif holdout == 'val':
                partition_dir = 'validation'
            else:
                partition_dir = holdout

            self.collect_all_java_files(
                join(self.data_dir, "java-med", partition_dir),
                holdout,
                join(self.data_dir, "java-med-file-wise")
            )


def to_snake_case(label: str):
    splitted = label.split('|')
    ans = splitted[0]
    for w in splitted[1:]:
        ans += (w[0].upper() + w[1:])
    return ans


# def normalize_token(token: str):
    # clean_token = re.sub(r"(\\\\n|\s+|[\"',])", "", token)  # replace escaped newline, whitespaces, qapostrophies, commas
    # return re.sub(r"[^A-Za-z]", "", clean_token)

def normalize_label(label):
    subtokens = re.split(
        r"(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\s+", label.strip())
    ans = []
    for st in subtokens:
        nt = re.sub(r"[^A-Za-z]", "", st.lower())
        if nt:
            ans.append(nt)
    return '|'.join(ans)


def contains_lbrace(ast):
    for node in ast:
        if node['token'] == '{':
            return True
    return False


def merge_source_ast():
    for holdout in ['train', 'val', 'test']:
        all_file_methods = {}
        print(f"Processing {holdout} part...")
        
        filelist = os.listdir('./java-med-file-wise/' + holdout)
        
        if os.path.isfile(holdout + '_chosen.pkl'):
            with open(holdout + '_chosen.pkl', 'rb') as f:
                chosen = pickle.load(f)
        else:
            chosen = random.sample(filelist, int(len(filelist) * (0.1 if holdout == 'train' else 0.05)))
            with open(holdout + '_chosen.pkl', 'wb') as f:
                pickle.dump(chosen, f)

        i = 0
        file_counter = 0
        files_total = len(filelist)
        for file_name in filelist:
            file_counter += 1
            try:
                with open('./java-med-file-wise/' + holdout + '/' + file_name, 'r') as fsc:
                    source_code = ''.join(fsc.readlines())
                    found = regex.findall(
                        r"((?:public|protected|private)\s)?"
                        r"((?:static\s)?(?:final\s)?(?:synchronized\s))?"
                        r"((?:native|strictfp)\s)?"
                        r"([\/\*\w<>\[\]\?,_\. ]+)"
                        r"(?<!new|if|for|while|\s)(\s+)"
                        r"([a-zA-Z\$_][\w\$]*)"
                        r"(\s*\((?:(?:[^\(\)]+\([^\(\)]*?\))??[^\(\)]*)*?\)[\w\s\.,@\/]*\{)",
                        source_code)
                    if not found:
                        continue

                    method_name = None
                    for method in found:
                        i += 1
                        mn = method[5]
                        if method_name is None:
                            method_name = mn
                        occ = (''.join(method)
                            .replace('\\', '\\\\')
                            .replace('[', '\[')
                            .replace(']', '\]')
                            .replace('(', '\(')
                            .replace(')', '\)')
                            .replace('{', '\{')
                            .replace('}', '\}')
                            .replace('.', '\.')
                            .replace('?', '\?')
                            .replace('*', '\*')
                            .replace('+', '\+')
                            .replace('|', '\|')
                            .replace('"', '\"')
                            .replace("'", "\'")
                            .replace('$', '\$')
                            .replace('^', '\^'))
                        if mn in all_file_methods:
                            all_file_methods[mn].append(occ)
                        else:
                            all_file_methods[mn] = [occ]

                        if method_name not in all_file_methods:
                            continue
                        try:
                            correct_occurrence = all_file_methods[method_name][0]
                        except IndexError:
                            continue

                        finally_found = re.search(correct_occurrence, source_code)
                        if not finally_found:
                            continue

                        start_pos = finally_found.start()

                        method_code = ''
                        stack = 0

                        sc_lines = source_code[start_pos:].split('\n')
                        for line in sc_lines:
                            if line == '':
                                continue
                            method_code += (line.strip('\n ').replace(method_name,
                                            'xxxmethodnamexxx') + ' ')
                            for brace in re.findall(r'[{}]', line):
                                first_found = True
                                if brace == '{':
                                    stack += 1
                                else:
                                    stack -= 1
                            if first_found and stack == 0:
                                break

                        method_code = re_0001_.sub(
                            re_0002, method_code).lower().strip()
                        method_code = re.sub(r'\s+', '|', method_code)

                        if method_code == '':
                            continue

                        example = {}
                        example['method_name_tokens'] = normalize_label(
                            method_name).split('|')
                        example['code_tokens'] = method_code.split('|')

                        os.makedirs(
                            "./optimization-methods/codebert/dataset/java-med", exist_ok=True)
                        filename = "./optimization-methods/codebert/dataset/java-med/" + holdout + '.jsonl'

                        if os.path.exists(filename):
                            append_write = 'a'
                        else:
                            append_write = 'w'

                        ff = open(filename, append_write)
                        ff.write(str(example) + '\n')
                        ff.close()

                        if i % 5000 == 0:
                            print('good', i)
                            print(file_counter, '/', files_total)

            except UnicodeDecodeError as e:
                print(e)
                print(i)
                continue


if __name__ == '__main__':
    # preprocessor = DataPreprocessor()
    # preprocessor.preprocess()
    merge_source_ast()
