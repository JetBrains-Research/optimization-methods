import argparse
import os
import shutil
import random
import pickle


parser = argparse.ArgumentParser(description='Train CodeBERTa.')
parser.add_argument('dir', type=str,
                    help='Path to directory with java-med to process.')
parser.add_argument('--train_divider', type=int, default=10,
                    help='Number of parts to split train and choose random one.')
parser.add_argument('--val_divider', type=int, default=5,
                    help='Number of parts to split val and choose random one.')
parser.add_argument('--random_idx_sets_count', type=int, default=5,
                    help='Number of partitions generated.')
args = parser.parse_args()

DIR = args.dir
COUNT = args.random_idx_sets_count
postfix = f'-train:{args.train_divider}-val:{args.val_divider}'
new_dir = f'java-med{postfix}/'

if DIR[-1] != '/':
    DIR += '/'

if os.path.isdir(new_dir):
    print('Dataset have already been processed.')
    quit()

with open(DIR + 'java-med.train.c2s') as f:
    lines_train = f.readlines()


length_train = len(lines_train)
print("train size:", length_train)

with open(DIR + 'java-med.val.c2s') as f:
    lines_val = f.readlines()

length_val = len(lines_val)
print("val size:", length_val)


files = list(filter(lambda x: 'idx' in x, os.listdir('.')))

os.mkdir(new_dir)

if len(files) == 0:
    for i in range(1, COUNT+1):
        idx_train = random.sample(
            range(length_train), k=length_train // args.train_divider)
        idx_val = random.sample(
            range(length_val), k=length_val // args.val_divider)
        print("mixing", './idx.' + str(i) + '.pickle')
        with open('idx.' + str(i) + '.pickle', 'wb') as f:
            pickle.dump((idx_train, idx_val), f)

    files = list(filter(lambda x: 'idx' in x, os.listdir('.')))

for file in files:
    num = file.split('.')[1]
    with open('idx.' + num + '.pickle', 'rb') as f:
        (idx_train, idx_val) = pickle.load(f)

    print("generating", new_dir + f'java-med{postfix}.train.' + num + '.c2s')
    with open(new_dir + f'java-med{postfix}.train.' + num + '.c2s', 'a') as out:
        for idx in idx_train:
            out.write(lines_train[idx])

    print("generating", new_dir + f'java-med{postfix}.val.' + num + '.c2s')
    with open(new_dir + f'java-med{postfix}.val.' + num + '.c2s', 'a') as out:
        for idx in idx_val:
            out.write(lines_val[idx])

shutil.copy2(DIR + 'java-med.test.c2s', new_dir)
shutil.copy2(DIR + 'java-med.dict.c2s', new_dir)

os.rename(new_dir + 'java-med.test.c2s', new_dir + f'java-med{postfix}.test.c2s')
os.rename(new_dir + 'java-med.dict.c2s', new_dir + f'java-med{postfix}.dict.c2s')