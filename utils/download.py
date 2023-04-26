import argparse
import pathlib as pl
import urllib.request as req
import shutil
import pandas as pd

from tqdm import tqdm

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
                df.drop(labels=i, axis=0, inplace=True)
                continue
            else:
                out_file = open(file_path, 'wb')
                shutil.copyfileobj(response, out_file)
        df.loc[i, 'img_url'] = file_path
    print("Process finished. Found {} errors".format(errors))
    df.to_json(json_dir / '{}.json'.format(split), orient='index', default_handler=str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloader')
    parser.add_argument('json_path', default=None, type=str,
                      help='source of the json dataset (default: None)')
    parser.add_argument('split', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('dst_path', default=None, type=str,
                      help='destination path (default: None)')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    download_images(args.json_path,args.split,args.dst_path)