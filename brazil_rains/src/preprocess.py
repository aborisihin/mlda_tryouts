#!/usr/bin/python3

import os.path

from datetime import datetime

DATA_FILE = './../data/project_brazil/sudeste.csv'
OUT_DIR = './../data/project_brazil/processed'


def preprocess(data_file, out_dir):

    with open(data_file, 'r') as in_file:

        time_start = datetime.now()
        total_lines = 0

        provinces_dict = dict()
        header_line = ''
        
        print('strings processed:', end=' ', flush=True)

        # read full set
        for line in in_file:

            total_lines += 1

            # store dataset header line
            if total_lines == 1:
                header_line = line
                continue # pass header string

            # progress indication
            if total_lines % 100000 == 0:
                print('{}k'.format(total_lines // 1000), end=' ', flush=True)

            # serch 'prov' field
            prov_name = line.split(',')[7]

            if not prov_name:
                continue
            
            # init dictionary element if prov value is new            
            if prov_name not in provinces_dict:
                prov_filepath = os.path.join(out_dir, 'prov_{}.csv'.format(prov_name))
                provinces_dict[prov_name] = {'file_obj': open(prov_filepath, 'w'), 'lines': 0}
                provinces_dict[prov_name]['file_obj'].write(header_line)

            # process input string
            provinces_dict[prov_name]['lines'] += 1
            provinces_dict[prov_name]['file_obj'].write(line)

        # close provinces files
        for prov in provinces_dict.values():
            prov['file_obj'].close()

        time_finish = datetime.now()
        process_time = time_finish - time_start

        print()
        print("process '{}' in total sec: {}".format(data_file, process_time.seconds))
        print('total lines (with header): {}'.format(total_lines))
        print('provinces lines:')
        for key, val in provinces_dict.items():
            print('{}: {}'.format(key, val['lines']))


if __name__ == '__main__':
    preprocess(DATA_FILE, OUT_DIR)
