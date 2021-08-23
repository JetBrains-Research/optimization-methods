import pickle


with open('./outputs/global_methods.data', 'rb') as f:
    stats = pickle.load(f)

methods = sorted(list(stats.keys()), key=lambda method: stats[method]['score']['chrF'], reverse=True)

string_output = " & ".join([" "*10] + ["{:7s}".format(method) for method in methods]) + "\n"

for metric in sorted(list(stats.items())[0][1]['score'].keys()):
    string_output += " & ".join(
        ["{:10s}".format(metric)] +\
        ["{:7.3f}".format(round(stats[method]['score'][metric], 3)) for method in methods]
        ) + "\n"

print(string_output)

text_file = open("table_code2seq.txt", "wt")
text_file.write(string_output)
text_file.close()
