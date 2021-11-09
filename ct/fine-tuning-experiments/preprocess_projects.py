import os
import json
import re
from shutil import rmtree, copytree, copy
import subprocess
from argparse import ArgumentParser


def get_sample_code(sample) -> str:
    return "".join(sample["code"].split())


def mask_recursive_calls(sample):
    mask = "METHOD_NAME"
    code = sample["code"]
    name = sample["name"]
    length = len(name)
    calls = [m.start() for m in re.finditer(f"[^a-zA-Z0-9_]{name}[^a-zA-Z0-9_]", code)]
    new_code = code
    delta = 0
    for i in range(1, len(calls)):
        start = calls[i] + 1 + delta
        new_code = new_code[:start] + mask + new_code[start + length:]
        delta += len(mask) - length
    sample["code"] = new_code
    return sample


def process_jsons() -> None:
    work_path = os.path.join("data", "raw", "code2seq-methods", "java-small")
    train_path = os.path.join(work_path, "train", "dataset-0.json")
    valid_path = os.path.join(work_path, "valid", "dataset-0.json")
    test_path = os.path.join(work_path, "test", "dataset-0.json")

    train = json.load(open(train_path, "r"))
    valid = json.load(open(valid_path, "r"))
    test = json.load(open(test_path, "r"))

    train_codes = set(get_sample_code(method) for method in train)
    valid_codes = set()
    new_valid = []
    for sample in valid:
        code = get_sample_code(sample)
        if code not in train_codes:
            valid_codes.add(code)
            new_valid.append(sample)

    new_test = []
    for sample in test:
        code = get_sample_code(sample)
        if code not in train_codes and code not in valid_codes:
            new_test.append(sample)

    train = [mask_recursive_calls(sample) for sample in train]
    with open(train_path, "w") as f:
        json.dump(train, f)

    valid = [mask_recursive_calls(sample) for sample in new_valid]
    with open(valid_path, "w") as f:
        json.dump(valid, f)

    test = [mask_recursive_calls(sample) for sample in new_test]
    with open(test_path, "w") as f:
        json.dump(test, f)

    stats = {"train": len(train), "val": len(valid), "test": len(test)}
    with open(os.path.join("data", "stage2", "stats.json"), "w") as f:
        json.dump(stats, f)


def process_single(project_name: str) -> None:
    print(f"Processing {project_name}...")
    rmtree("data")
    raw_dataset = os.path.join("data", "raw", "code2seq", "java-small")

    os.makedirs(raw_dataset)
    os.mkdir(os.path.join("data", "stage1"))
    os.mkdir(os.path.join("data", "stage2"))

    project_path = os.path.join("raw_java", project_name)
    for part in zip(["train", "val", "test"], ["training", "validation", "test"]):
        copytree(os.path.join(project_path, part[0]), os.path.join(os.path.join(raw_dataset, part[1])))

    cmd = "python -m scripts.extract-java-methods java-small"
    subprocess.check_call(cmd, shell=True)

    process_jsons()

    for part in ["train", "valid", "test"]:
        cmd = (
            f"python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-code2seq"
            f".yaml java-small {part} "
        )
        subprocess.check_call(cmd, shell=True)

    stage2_path = os.path.join("data", "stage2", "java-small")
    stage1_path = os.path.join("data", "stage1", "java-small")

    os.makedirs(stage2_path)
    copy(os.path.join("binaries", "vocabularies.p.gzip"), stage2_path)
    copytree(os.path.join(stage1_path, "valid"),
             os.path.join(stage1_path, "real_valid"))
    rmtree(os.path.join(stage1_path, "valid"))
    copytree(os.path.join(stage1_path, "train"),
             os.path.join(stage1_path, "valid"))
    cmd = (
        f"python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2.yaml "
        f"java-small valid"
    )
    subprocess.check_call(cmd, shell=True)

    copytree(os.path.join(stage2_path, "valid"),
             os.path.join(stage2_path, "train"))
    rmtree(os.path.join(stage1_path, "valid"))
    rmtree(os.path.join(stage2_path, "valid"))
    copytree(os.path.join(stage1_path, "real_valid"),
             os.path.join(stage1_path, "valid"))
    rmtree(os.path.join(stage1_path, "real_valid"))

    for part in ["valid", "test"]:
        cmd = (
            f"python -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2.yaml "
            f"java-small {part} "
        )
        subprocess.check_call(cmd, shell=True)

    preprocessed_path = os.path.join("preprocessed", project_name, "stage2")
    copytree(os.path.join("data", "stage2"), preprocessed_path)

    print("Finished!")


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("project_names", type=str, help="Path to file with project names")
    args = arg_parser.parse_args()

    with open(args.project_names, "r") as f:
        for name in f:
            process_single(name.strip())
