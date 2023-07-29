import argparse
import json
import numpy as np
from torchvision import datasets, transforms


def load_labels(filename):
    with open(filename) as labels_file:
      labels_raw = json.load(labels_file)

    labels_raw= [label for label in labels_raw if label['newLabel'] not in "psud-"] # remove non-integer input
    labels = {label['index']: int(label['newLabel']) for label in labels_raw} # remove duplicates, keeping last
    return labels

def print_confusion(array2d):
    print()
    def print_row(first_column, row):
      print(first_column, end='')
      for x in row:
        print(f'{x:5d}', end='')
      print()
    print_row("   |", range(array2d.shape[1]))
    print('-' * ( 5 * array2d.shape[1] + 4))
    for i, row in enumerate(array2d):
      print_row(f"{i:2d} |", row)
    print()


def report_statistics(original_labels, labels):
    print(f'nr of labels {len(labels)}')

    nr_correct = sum(1 for index in labels if original_labels[index] == labels[index])
    print(f'percent equal to original label: {100 * nr_correct / len(labels):.3f}')

    confusion = np.zeros((10, 10), dtype=int)
    for index in labels:
      confusion[original_labels[index], labels[index]] += 1
    print_confusion(confusion)


def main():
    parser = argparse.ArgumentParser(description='Print some statistics for annotation file')
    parser.add_argument("annotations_filename", type=str, help='file with annotations')
    parser.add_argument("--original_labels_filename", type=str, default='', help='file with original labels')
    args = parser.parse_args()
    if args.original_labels_filename:
        with open(args.original_labels_filename) as original_labels_file:
            original_labels = json.load(original_labels_file)['labels']
    else:
        dataset_train = datasets.MNIST('../data', train=True, download=True)
        original_labels = [sample[1] for sample in dataset_train]
    labels = load_labels(args.annotations_filename)
    report_statistics(original_labels, labels)


if __name__ == '__main__':
    main()
