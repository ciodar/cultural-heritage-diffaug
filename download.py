import argparse
import pathlib as pl
import urllib.request as req
import shutil
import pandas as pd

from tqdm import tqdm


def download_images(json_path, dst_path, user: str):
    """
    This script downloads the images from the urls in the json file and saves them in the destination folder.
    @param json_path: path to the json file containing the urls of the images to download
    @param dst_path: path to the folder where the images will be saved
    """
    # read the json file we want to download images from
    json_path = pl.Path(json_path)
    json_dir = pl.Path(json_path).parent
    with open(json_path, 'r') as f:
        data = pd.read_json(f, orient='index')
    # get the selected split from the json
    dst_path = pl.Path(dst_path)
    dst_path.mkdir(parents=False, exist_ok=True)
    errors = 0

    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        url = row['img_url']
        filename = pl.Path(url).name
        file_path = dst_path / filename
        if not pl.Path(file_path).exists():
            request = req.Request(url)
            request.add_header('User-Agent', f'CulturalHeritageBot/0.0 ({user})')
            try:
                response = req.urlopen(request)
                out_file = open(file_path, 'wb')
                shutil.copyfileobj(response, out_file)
            except Exception as e:
                errors += 1
                print("Cannot find {}, error code: {}".format(url, e.code))
                data.drop(index=i, inplace=True)
                continue
        data.loc[i, 'img_path'] = f"{dst_path.name}/{filename}"
    print("Process finished. Found {} errors".format(errors))
    data.to_json(json_dir / '{}_local.json'.format(json_path.stem), orient='index', default_handler=str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloader')
    parser.add_argument('json_path', default=None, type=str,
                        help='source of the json dataset (default: None)')
    parser.add_argument('dst_path', default=None, type=str,
                        help='destination path (default: None)')
    parser.add_argument('user', type=str,
                        help='identifier for the user agent')
    args = parser.parse_args()
    # custom cli options to modify configuration from default values given in json file.
    download_images(args.json_path, args.dst_path, args.user)
