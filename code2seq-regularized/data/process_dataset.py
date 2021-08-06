import os
import random
import pickle

COUNT = 5
DIR = input('enter dir (e.g., ./java-med/ or default): ')

if "default" in DIR.lower():
    DIR = './java-med/'

if DIR[-1] != '/':
    DIR += '/'
    
with open(DIR + 'java-med.train.c2s') as f:
    lines_train = f.readlines()


length_train = len(lines_train)
print("train size:", length_train)

with open(DIR + 'java-med.val.c2s') as f:
    lines_val = f.readlines()

length_val = len(lines_val)
print("val size:", length_val)


files = list(filter(lambda x: 'idx' in x, os.listdir('.')))

if len(files) == 0:
    for i in range(1, COUNT+1):
        idx_train = random.sample(range(length_train), k=length_train // 10)
        idx_val = random.sample(range(length_val), k=length_val // 5)
        print("mixing", './idx.' + str(i) + '.pickle')
        with open('idx.' + str(i) + '.pickle', 'wb') as f:
            pickle.dump((idx_train, idx_val), f)
    
    files = list(filter(lambda x: 'idx' in x, os.listdir('.')))
    
for file in files:
    num = file.split('.')[1]
    with open('idx.' + num + '.pickle', 'rb') as f:
        (idx_train, idx_val) = pickle.load(f)

    print("generating", DIR + 'java-med.train.' + num + '.c2s')
    with open(DIR + 'java-med.train.' + num + '.c2s', 'a') as out:
        for idx in idx_train:
            out.write(lines_train[idx])

    print("generating", DIR + 'java-med.val.' + num + '.c2s')
    with open(DIR + 'java-med.val.' + num + '.c2s', 'a') as out:
        for idx in idx_val:
            out.write(lines_val[idx])
