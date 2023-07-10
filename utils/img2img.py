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


def main(args):
    input_json = args.input_json
    input_folder = pl.Path(args.input_folder)
    output_folder = pl.Path(args.output_folder)
    n_variations = args.n_variations

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

    for i, (id, item) in enumerate(tqdm(data.items())):
        # print(item)
        image_path = input_folder / item["img_path"]
        prompt = item["caption"][0]
        if item["split"] != "train":
            continue

        folder_name = f"{id}_{pl.Path(image_path).stem}"
        (output_folder / folder_name).mkdir(exist_ok=True)

        assert image_path.exists(), f"Image path {image_path} does not exist"
        with Image.open(image_path) as im:
            size = im.size
            buffered = BytesIO()
            im.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        width, height = size
        if width > height:
            width = DIMENSION
            height = int(DIMENSION * size[1] / size[0])
        else:
            height = DIMENSION
            width = int(DIMENSION * size[0] / size[1])


        for s in range(n_variations):
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
            }

            img2img_response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/img2img', json=payload)

            print(output_folder / folder_name /  f"{s:05d}_{steps}_{ALIASES[sampler_name]}_{s}_0.png")
            ## Check status
            if img2img_response.status_code != 200:
                print(img2img_response.text)
                break
            else:
                r = img2img_response.json()
                image = r["images"][0]
                image = Image.open(BytesIO(base64.b64decode(image)))
                image.save(os.path.join(output_folder, folder_name, f"{s:05d}_{steps}_{ALIASES[sampler_name]}_{s}_0.png"))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="C:\\path\\to\\input\\folder")
    parser.add_argument("--input_folder", type=str, default="C:\\path\\to\\input\\folder")
    parser.add_argument("--output_folder", type=str, default="C:\\path\\to\\output\\folder")
    parser.add_argument("--n_variations", type=int, default=1)
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1:7860")
    parser.add_argument("--cfg_scale", type=int, default=10)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sampler_name", type=str, default="Euler a")

    main(parser.parse_args())