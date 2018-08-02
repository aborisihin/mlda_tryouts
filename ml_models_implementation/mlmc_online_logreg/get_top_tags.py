#!/usr/bin/python

import sys
import getopt
import os.path

from collections import Counter


def main(argv):

    data_dir = None
    input_file = None
    output_file = None
    tags_count = None

    try:
        opts, _ = getopt.getopt(argv, 'd:i:o:n:')
    except getopt.GetoptError:
        notify_error()
        sys.exit(2)

    for (opt, arg) in opts:
        if opt == '-d':
            data_dir = arg
        elif opt == '-i':
            input_file = arg
        elif opt == '-o':
            output_file = arg
        elif opt == '-n':
            tags_count = int(arg)

    if data_dir is None:
        data_dir = './'

    if tags_count is None:
        tags_count = 10

    if (input_file is not None) & (output_file is not None):
        get_top_tags(data_dir, input_file, output_file, tags_count)
    else:
        notify_error()
        sys.exit(2)


def notify_error():

    print('usage: get_top_tags.py [-d <data_dir>] -i <input_file> -o <output_file> [-n <tags_count> default=10]')
    sys.exit(2)


def get_top_tags(data_dir, input_file, output_file, tags_count):

    in_filepath = os.path.join(data_dir, input_file)
    out_filepath = os.path.join(data_dir, output_file)

    tags_counter = Counter()

    try:
        with open(in_filepath, 'r') as in_file:
            for line in in_file:
                _, tags = line.strip().split('\t')
                tags_counter.update(tags.split(' '))
    except EnvironmentError:
        print('Error opening file {}'.format(in_filepath))
        exit(2)

    top_tags = [tc[0] for tc in tags_counter.most_common(tags_count)]

    try:
        with open(out_filepath, 'w') as out_file:
            for tag in top_tags:
                out_file.write(tag + '\n')
    except EnvironmentError:
        print('Error opening file {}'.format(out_filepath))
        exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
