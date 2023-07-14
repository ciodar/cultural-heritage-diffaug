import argparse
import base64
from io import BytesIO
import os
import requests
from PIL import Image
import json
import pathlib as pl
from tqdm import tqdm

ALIASES = {
    "Euler a": "k_euler_a",
}
DIMENSION = 576
MAX_SIZE = 4096 * 2
# avoids PIL.Image.DecompressionBombError
# https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
Image.MAX_IMAGE_PIXELS = None

def main(args):
    input_json = args.input_json
    input_folder = pl.Path(args.input_folder)
    output_folder = pl.Path(args.output_folder)
    api_url = args.api_url
    n_variations = args.n_variations
    batch_size = args.batch_size

    # Diffusion params
    steps = args.steps
    sampler_name = args.sampler_name
    cfg_scale = args.cfg_scale
    #

    assert input_folder.exists(), f"Input folder {input_folder} does not exist"
    assert output_folder.parent.exists(), f"Output folder {output_folder.parent} does not exist"
    output_folder.mkdir(exist_ok=True)

    with open(input_json, "r") as f:
        data = json.load(f)

    folders = {f.name.split('_')[0]: f.name for f in output_folder.iterdir() if f.is_dir()}

    for i, (id, item) in enumerate(tqdm(data.items())):
        # print(item)
        image_path = input_folder / item["img_path"]
        prompt = item["caption"][0]
        title = image_path.stem if item.get("title") is None else item["title"]

        if item["split"] != "train":
            continue
        if id in folders:
            folder_name = folders[id]
        else:
            folder_name = f"{id}_{title}"

        if (output_folder / folder_name).exists() and len(
                list((output_folder / folder_name).iterdir())) >= n_variations:
            #print(f"Skipping {folder_name} as it already exists")
            continue
        else:
            (output_folder / folder_name).mkdir(exist_ok=True)

        assert image_path.exists(), f"Image path {image_path} does not exist"
        with Image.open(image_path) as im:
            size = im.size
            width, height = size
            if width > height:
                width = DIMENSION
                height = int(DIMENSION * size[1] / size[0])
            else:
                height = DIMENSION
                width = int(DIMENSION * size[0] / size[1])
            if max(size) > MAX_SIZE:
                im.thumbnail((width, height), Image.LANCZOS)
            buffered = BytesIO()
            im.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        for s in range(0, n_variations, batch_size):
            payload = {
                "init_images": [img_base64],
                "prompt": prompt,
                "denoising_strength": 0.34,
                "width": width,
                "height": height,
                "cfg_scale": cfg_scale,
                "sampler_name": sampler_name,
                "restore_faces": False,
                "steps": steps,
                "seed": s,
                "batch_size": batch_size,
            }

            img2img_response = requests.post(url=f'{api_url}/sdapi/v1/img2img', json=payload)

            ## Check status
            if img2img_response.status_code != 200:
                print(img2img_response.text)
                raise Exception(f"Error in img2img API call: {img2img_response.status_code}")
            else:
                r = img2img_response.json()
                for i, image in enumerate(r["images"]):
                    image = Image.open(BytesIO(base64.b64decode(image)))
                    # print(os.path.join(output_folder, folder_name, f"{s+i:05d}_{steps}_{ALIASES[sampler_name]}_{s+i}_0.png"))
                    image.save(os.path.join(output_folder, folder_name,
                                            f"{s + i:05d}_{steps}_{ALIASES[sampler_name]}_{s + i}_0.png"))
        #break # DEBUG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="C:\\path\\to\\input\\folder")
    parser.add_argument("--input_folder", type=str, default="C:\\path\\to\\input\\folder")
    parser.add_argument("--output_folder", type=str, default="C:\\path\\to\\output\\folder")
    parser.add_argument("--n_variations", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:7860")
    parser.add_argument("--cfg_scale", type=int, default=10)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sampler_name", type=str, default="Euler a")

    main(parser.parse_args())
