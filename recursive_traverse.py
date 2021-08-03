from os import listdir
from os.path import join, isfile
import shutil
import json
import re
import regex


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

DATA_DIR = "/home/anton/Java-med"


class DataPreprocessor:
    def __init__(self, data_dir: str = "/home/anton/Java-med"):
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
                    shutil.copyfile(
                        cur_path,
                        join(copy_dir, holdout, f"{self.indices[holdout]}.java")
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
    subtokens = re.split(r"(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\s+", label.strip())
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
        print("Processing {holdout} part...")
        with open(f"{DATA_DIR}/prepreprocessed/{holdout}-asts/java/asts.jsonl", 'r') as fin,\
                open(f"{DATA_DIR}/java-med-source-asts/java-med-source-asts.{holdout}.jsonl", 'w') as fout,\
                open(f"{DATA_DIR}/prepreprocessed/{holdout}-bad-samples.txt", 'w') as ferr:
            cur_file_methods_occs = (-1, {})
            for line in fin:
                example = json.loads(line)
                if not contains_lbrace(example['ast']):  # empty methods filtration
                    continue
                file_name = example.pop('origFile')
                method_name = example['label'].replace('$', '\$')  # the only special character allowed in method name
                if cur_file_methods_occs[0] != file_name:
                    cur_file_methods_occs = file_name, {}
                if method_name in cur_file_methods_occs[1]:
                    cur_file_methods_occs[1][method_name] += 1
                else:
                    cur_file_methods_occs[1][method_name] = 0
                method_code = ''
                stack = 0
                first_found = False
                with open(file_name, 'r') as fsc:
                    try:
                        source_code = ''.join(fsc.readlines())
                        found = regex.findall(
                            rf"((?:public|protected|private)\s)?((?:static\s)?(?:final\s)?(?:synchronized\s))?((?:native|strictfp)\s)?([\/\*\w<>\[\]\?,_\. ]+)(?<!new|if|return)(\s+{method_name}"
                            r"\s*\((?:[^\)]+(?:\(.*?\))?[^\)]*)*\)[\w\s\.,]*\{)", source_code)
                        if found:
                            try:
                                correct_occurrence = found[cur_file_methods_occs[1][method_name]]
                            except IndexError:
                                ferr.write(f"{file_name} - {method_name} - wrong methods mapping\n")
                                continue

                            correct_occurrence = (''.join(correct_occurrence)
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
                                                            .replace('^', '\^')
                                                            .replace('\\$', '\$'))  # in case of double screening in method_name

                            finally_found = re.search(correct_occurrence, source_code)

                            try:
                                start_pos = finally_found.start()
                            except AttributeError:
                                ferr.write(f"{file_name} - {method_name} - secondary error\n")
                                continue

                            sc_lines = source_code[start_pos:].split('\n')
                            for line in sc_lines:
                                if line == '':
                                    continue
                                method_code += (line.strip('\n ').replace(method_name, 'xxxmethodnamexxx') + ' ')
                                for brace in re.findall(r'[{}]', line):
                                    first_found = True
                                    if brace == '{':
                                        stack += 1
                                    else:
                                        stack -= 1
                                if first_found and stack == 0:
                                    break
                    except UnicodeDecodeError:
                        ferr.write(f"{file_name} - file decodeing error")
                        continue
                method_code = re_0001_.sub(re_0002, method_code).lower().strip()
                method_code = re.sub(r'\s+', '|', method_code)
                # print(file_name, method_name, method_code, sep='\n')
                # return
                if method_code == '':
                    ferr.write(f"{file_name} - {method_name}\n")
                    continue
                example['label'] = normalize_label(method_name)
                example['SOURCE'] = method_code
                example['AST'] = example.pop('ast')
                for node in example['AST']:
                    node['node'] = node.pop('typeLabel')
                json_obj = json.dumps(example)
                fout.write(str(json_obj) + '\n')


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    preprocessor.preprocess()
    merge_source_ast()
    # print(normalize_label("is4Very_Empty21"))
