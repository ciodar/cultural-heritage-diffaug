import json

import lightning as L
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pathlib as pl
import torch
import random
from transformers import AutoProcessor

from data_loader.example import Example

# avoids PIL.Image.DecompressionBombError
# https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
Image.MAX_IMAGE_PIXELS = None

""""
Dataset class for Image Captioning on Artpedia dataset (https://iris.unimore.it/retrieve/handle/11380/1178736/224456/paper.pdf)

"""


class ArtpediaDataset(Dataset):
    def __init__(self, samples: list[Example], transform=None, processor=None, captions_per_image=1
                 , caption_mode='random', sd_augmentation=0.0):
        self.data = samples
        self.transform = transform
        self.processor = processor
        self.captions_per_image = captions_per_image
        self.caption_mode = caption_mode
        self.sd_augmentation = sd_augmentation

    def _get_image(self, images):
        if random.random() < self.sd_augmentation and len(images) > 1:
            # Choose randomly one of the augmented images
            image_path = random.choice(images[1:])
        else:
            # Choose the original image
            image_path = images[0]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def collate_fn(self, batch):
        images, texts, gts = zip(*batch)
        encoded = self.processor(text=texts, images=images, padding="max_length",
                                 truncation=True, return_tensors="pt")
        encoded['labels'] = encoded['input_ids']

        return encoded, gts

    def __getitem__(self, i):
        sample = self.data[i]
        image = self._get_image(sample.images)
        id = sample.id
        gts = sample.captions
        caption = random.choice(gts)
        return image, caption, {id: gts}

    def __len__(self):
        return len(self.data)


class ArtpediaDataModule(L.LightningDataModule):
    def __init__(self,
                 img_dir: str = './data/artpedia',
                 ann_file: str = './data/artpedia/artpedia.json',
                 batch_size: int = 2,
                 model_name_or_path: str = None,
                 caption_mode: str = 'first',
                 captions_per_image: int = 1,
                 sd_augmentation=0.0,
                 num_workers: int = 1,
                 transform=None):
        super().__init__()
        self.img_dir = pl.Path(img_dir)
        self.ann_file = ann_file
        # set captioning mode
        self.captions_per_image = captions_per_image
        self.caption_mode = caption_mode
        self.sd_augmentation = sd_augmentation
        # set model name for processor
        self.model_name_or_path = model_name_or_path
        # for now use same batch size for train and test
        self.batch_size = batch_size
        self.num_workers = num_workers
        # TODO: research augmentation for Image Captioning / VQA
        # TODO: Check why Resize crops image
        self.processor = None
        self.transform = None
        if transform:
            self.transform = []
            for t in transform:
                # module_name, classpath = t['class_path'].split('.')[0], '.'.join(t['class_path'].split('.')[1:])
                # module = importlib.import_module(module_name)
                # args = t.get('init_args', {})
                # t_obj = rgetattr(module, classpath)(**args)
                self.transform.append(t)
            self.transform = torch.nn.Sequential(*self.transform)
        #if self.transform is not None:


    def prepare_data(self):
        AutoProcessor.from_pretrained(self.model_name_or_path)

    def setup(self, stage: str) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        train_samples, val_samples, test_samples = [], [], []
        with open(self.ann_file, 'r') as f:
            self.data = json.load(f)
            self.ids = list(self.data.keys())
        for k, v in self.data.items():
            captions = v['caption']
            if len(captions) < self.captions_per_image:
                captions = captions + [random.choice(captions)
                                       for _ in range(self.captions_per_image - len(captions))]
            filename = v['img_path']
            if self.sd_augmentation:
                augmented_images = v.get('sd_augmentations', [])
            else:
                augmented_images = []
            example = Example.fromdict({
                'id': k,
                'images': [self.img_dir / filename] + [self.img_dir / filename for filename in augmented_images],
                'captions': captions,
            })
            if v['split'] == 'train':
                train_samples.append(example)
            elif v['split'] == 'val':
                val_samples.append(example)
            elif v['split'] == 'test':
                test_samples.append(example)

        if stage == "fit":
            self.train_ds = ArtpediaDataset(train_samples, transform=self.transform
                                            , processor=self.processor, captions_per_image=1
                                            , caption_mode='first', sd_augmentation=self.sd_augmentation)
            self.valid_ds = ArtpediaDataset(val_samples, transform=None
                                            , processor=self.processor, captions_per_image=self.captions_per_image
                                            , caption_mode=self.caption_mode)
        if stage == "validate":
            self.valid_ds = ArtpediaDataset(val_samples, transform=None
                                            , processor=self.processor, captions_per_image=self.captions_per_image
                                            , caption_mode=self.caption_mode)
        if stage == "test":
            self.test_ds = ArtpediaDataset(test_samples, transform=None
                                           , processor=self.processor, captions_per_image=self.captions_per_image
                                           , caption_mode=self.caption_mode)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.train_ds.collate_fn,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, collate_fn=self.valid_ds.collate_fn,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.test_ds.collate_fn,
                          num_workers=self.num_workers)
