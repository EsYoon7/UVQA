import json
import ipdb
import os
import sys
import argparse
import glob
import numpy as np
import torch
import random


def set_runname(args):
    from datetime import date, datetime, timezone, timedelta
    KST  = timezone(timedelta(hours=9))
    args.date = str(date.today())
    time_record = str(datetime.now(KST).time())[:8]

    args.run_name = args.date +'_'+time_record
    return args

def set_device(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cpu = torch.device("cpu")
    return args

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True
    

def set_output_setting(args, question_type="relation"):
    args.output_path = os.path.join(args.root_dir, "data_generation", question_type, args.run_name)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    args.output_fname = os.path.join(args.output_path, args.output_file_name+'.json')

    return args

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def load_txt(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return [item.strip() for item in data]

def set_unique_dict(dict_):
    for k, v in dict_.items():
        dict_[k] = list(set(v))
    return dict_

def get_list_from_dict(dict_):
    output_list = []
    for k, v in dict_.items():
        processed_list = [v_.replace('unclassified ', '') for v_ in v]
        output_list += processed_list
    return output_list

def get_list_from_dict_item(dict_, item):
    for k, v in dict_.items():
        if item in v:
            return v

def set_unique_list(list_):
    return list(set(list_))