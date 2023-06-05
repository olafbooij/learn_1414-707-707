import json
import numpy as np
from torchvision import datasets, transforms


def load_labels(filename):
    with open(filename) as labels_file:
      labels_raw = json.load(labels_file)
    labels_raw= [label for label in labels_raw if label['newLabel'] not in "psud-"] # remove non-integer input
    labels = {label['index']: label['newLabel'] for label in labels_raw} # remove duplicates, keeping last
    return labels


def report_statistics(dataset_orig, labels):
    print(f'nr of labels {len(labels)}')

    nr_correct = sum(1 for index in labels if dataset_orig[index][1] == int(labels[index]))
    print(f'part equal to original label: {nr_correct / len(labels)}')

    confusion = np.zeros((10, 10))
    for index in labels:
      confusion[dataset_orig[index][1]][int(labels[index])] += 1
    print(confusion)


def main():
    dataset_train = datasets.MNIST('../data', train=True, download=True)
    labels = load_labels('mnist_labels/test2894.json')
    report_statistics(dataset_train, labels)


if __name__ == '__main__':
    main()
