import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import pathlib as pl
import urllib.request as req
import shutil

from tqdm import tqdm


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def download_images(json_path, split, dst_path):
    # read the json file we want to download images from
    json_path = pl.Path(json_path)
    json_dir = pl.Path(json_path).parent
    with open(json_path, 'r') as f:
        data = pd.read_json(f, orient='index')
    # get the selected split from the json
    df = data[data.split == split].copy().reset_index()
    # get the download folder
    dst_path = pl.Path(dst_path)
    dst_path.mkdir(parents=False, exist_ok=True)
    errors = 0
    for i, url in enumerate(tqdm(df['img_url'])):
        filename = pl.Path(url).name
        file_path = dst_path / filename
        if not pl.Path(file_path).exists():
            request = req.Request(url)
            request.add_header('User-Agent', 'CulturalHeritageBot/0.0 (dario.cioni@stud.unifi.it)')
            try:
                response = req.urlopen(request)
            except Exception as e:
                errors += 1
                print("Cannot find {}, error code: {}".format(url, e.code))
                df.drop(labels=i, axis=0)
                continue
            else:
                out_file = open(file_path, 'wb')
                shutil.copyfileobj(response, out_file)
        df.loc[i, 'img_url'] = file_path
    print("Process finished. Found {} errors".format(errors))
    df.to_json(json_dir / '{}.json'.format(split), orient='index', default_handler=str)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
