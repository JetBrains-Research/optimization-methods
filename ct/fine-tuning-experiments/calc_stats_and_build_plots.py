import numpy as np
import matplotlib.pyplot as plt
import os
import json


def save_metric_plot(metric_name: str):
    metric_new = []
    metric_trained_before = []
    metric_trained_after = []
    for project in os.listdir("results"):
        project_folder = os.path.join("results", project)

        with open(os.path.join(project_folder, "metrics_new_after.json"), "r") as f:
            metric_new.append(json.load(f)[metric_name])

        with open(os.path.join(project_folder, "metrics_trained_before.json"), "r") as f:
            metric_trained_before.append(json.load(f)[metric_name])

        with open(os.path.join(project_folder, "metrics_trained_after.json"), "r") as f:
            metric_trained_after.append(json.load(f)[metric_name])

    metric_new = np.asarray(metric_new)
    metric_trained_before = np.asarray(metric_trained_before)
    metric_trained_after = np.asarray(metric_trained_after)

    fig, ax = plt.subplots()

    metric_name = metric_name.upper()
    ax.plot(np.arange(1, metric_new.shape[0] + 1), np.sort(metric_new), "o", label="From scratch")
    ax.plot(
        np.arange(1, metric_trained_before.shape[0] + 1), np.sort(metric_trained_before), "o", label="Pretrained"
    )
    ax.plot(np.arange(1, metric_trained_after.shape[0] + 1), np.sort(metric_trained_after), "o", label="Fine-tuned")
    ax.legend()

    plt.ylabel(metric_name)

    fig.suptitle(f"{metric_name} distribution", fontweight="bold")
    fig.savefig(os.path.join("images", f"{metric_name}.png"))

    print(f"{metric_name} mean improved", np.mean(metric_trained_after - metric_trained_before))
    print(f"{metric_name} mean from scratch", np.mean(metric_new))


if __name__ == "__main__":
    save_metric_plot("f1")
    save_metric_plot("bleu")
    save_metric_plot("chrf")
