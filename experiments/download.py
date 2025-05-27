# -*- coding: utf-8 -*-
# Author  : liyanpeng
# Email   : yanpeng.li@cumt.edu.cn
# Datetime: 2024/5/28 18:23
# Filename: download_data.py
from mteb import MTEB

import os
import subprocess

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
data_path = '/data1/jiyifan/hf_data'

def show_dataset():
    evaluation = MTEB(task_langs=["zh", "zh-CN"])
    dataset_list = []
    for task in evaluation.tasks:
        if task.description.get('name') not in dataset_list:
            dataset_list.append(task.description.get('name'))
            desc = 'name: {}\t\thf_name: {}\t\ttype: {}\t\tcategory: {}'.format(
                task.description.get('name'), task.description.get('hf_hub_name'),
                task.description.get('type'), task.description.get('category'),
            )
            print(desc)
    print(len(dataset_list))

def download_dataset():
    evaluation = MTEB(task_langs=["zh", "zh-CN"])
    err_list = []
    for task in evaluation.tasks:
        # task.load_data()
        # https://huggingface.co/datasets/
        task_name = task.description.get('hf_hub_name')
        print(task_name)
        cmd = ['huggingface-cli', 'download', '--repo-type', 'dataset', '--resume-download',
               '--local-dir-use-symlinks', 'False', task_name, '--local-dir', os.path.join(data_path, task_name)]
        try:
            result = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            err_list.append(task_name)
            print("{} is error".format(task_name))

    if err_list:
        print('download failed: \n', '\n'.join(err_list))
    else:
        print('download success.')

if __name__ == '__main__':
	download_dataset()
	show_dataset()
