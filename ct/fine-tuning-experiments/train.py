from argparse import ArgumentParser

from shutil import rmtree, copytree
from math import ceil
import os
import json
import subprocess
from omegaconf import OmegaConf
from .test_model import calculate_metrics


def modify_config(config_path: str, part_sizes) -> None:
    config = OmegaConf.load(open(config_path, "r"))
    config["training"]["simulated_batch_size_valid"] = part_sizes["val"]
    config["training"]["persistent_snapshot_every"] = ceil(
        part_sizes["train"] / config["training"]["batch_size"])
    OmegaConf.save(config, config_path)


def get_run_id_and_snapshot():
    models = os.path.join("models", "ct_code_summarization")
    names = os.listdir(models)
    model = sorted(names)[-1]
    snapshot = sorted(os.listdir(os.path.join(models, model)))[-1]
    snapshot = "".join(x for x in snapshot if x in "0123456789")
    return model, snapshot


def fine_tune_and_save_metrics(project_name: str) -> None:
    data_path = os.path.join("data", "stage2")
    rmtree(data_path)
    copytree(os.path.join("preprocessed", project_name, "stage2"), data_path)
    result_path = os.path.join("results", project_name)
    os.mkdir(result_path)

    part_sizes = json.load(open(os.path.join(data_path, "stats.json"), "r"))

    # config_path = os.path.join("fine-tuning-experiments", "from_scratch_config.yaml")
    # modify_config(config_path, part_sizes)
    # cmd = f"python -m scripts.run-experiment {config_path}"
    # subprocess.check_call(cmd, shell=True)
    model, snapshot = get_run_id_and_snapshot()
    metrics_new_after = calculate_metrics(model, snapshot)
    with open(os.path.join(result_path, "metrics_new_after.json"), "w") as file:
        json.dump(metrics_new_after, file)

    metrics_trained_before = calculate_metrics("CT-20", "30000")
    with open(os.path.join(result_path, "metrics_trained_before.json"), "w") as file:
        json.dump(metrics_trained_before, file)

    config_path = os.path.join("fine-tuning-experiments", "fine_tuning_config.yaml")
    modify_config(config_path, part_sizes)
    cmd = f"python -m scripts.run-experiment {config_path}"
    subprocess.check_call(cmd, shell=True)
    model, snapshot = get_run_id_and_snapshot()
    metrics_trained_after = calculate_metrics(model, snapshot)
    with open(os.path.join(result_path, "metrics_trained_after.json"), "w") as file:
        json.dump(metrics_trained_after, file)


def train_from_scratch():
    config_path = os.path.join("fine-tuning-experiments", "from_scratch_config.yaml")
    cmd = f"python3 -m scripts.run-experiment {config_path}"
    subprocess.check_call(cmd, shell=True)
    model, snapshot = get_run_id_and_snapshot()
    metrics_trained_after = calculate_metrics(model, snapshot)
    result_path = os.path.join("results", "full")
    os.mkdir(result_path)
    with open(os.path.join(result_path, "metrics_trained_after.json"), "w") as file:
        json.dump(metrics_trained_after, file)


if __name__ == "__main__":
    # arg_parser = ArgumentParser()
    # arg_parser.add_argument("project_names", type=str, help="Path to file with project names")
    # args = arg_parser.parse_args()

    # with open(args.project_names, "r") as f:
    #     for name in f:
    #         fine_tune_and_save_metrics(name.strip())
    train_from_scratch()
