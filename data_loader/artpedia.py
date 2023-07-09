import json

import lightning as L
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pathlib as pl
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
    def __init__(self, samples: list[Example], transform=None, processor=None, captions_per_image=10
                 , caption_mode='random', sd_augmentation=0.0):
        self.data = samples
        self.transform = transform
        self.processor = processor
        self.captions_per_image = captions_per_image
        self.caption_mode = caption_mode
        self.sd_augmentation = sd_augmentation

    def _get_captions(self, all_captions):
        if type(all_captions) == str:
            return all_captions, [all_captions]
        if len(all_captions) < self.captions_per_image:
            captions = all_captions + [random.choice(all_captions)
                                       for _ in range(self.captions_per_image - len(all_captions))]
        else:
            captions = random.sample(all_captions, k=self.captions_per_image)
        if self.caption_mode == 'random':
            caption = random.choice(all_captions)
        elif self.caption_mode == 'first':
            caption = all_captions[0]
        elif self.caption_mode == 'cat':
            caption = '\n'.join(all_captions)
            captions = [caption]
        else:
            raise ValueError('Caption mode {} not supported'.format(self.caption_mode))
        return caption, captions

    def _get_image(self, index):
        if random.random() < self.sd_augmentation and self.data[index].augmented_images:
            # Choose randomly one of the augmented images
            image_path = random.choice(self.data[index].augmented_images)
        else:
            # Choose the original image
            image_path = self.data[index].image
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def collate_fn(self, batch):
        images, captions = zip(*batch)
        encoded = self.processor(images=images, text=captions, padding="max_length",
                                 truncation=True, return_tensors="pt")
        encoded['labels'] = encoded['input_ids']
        return encoded

    def __getitem__(self, i):
        image = self._get_image(i)
        # get all visual sentences
        all_captions = self.data[i].text
        # get single caption and padded list of captions
        caption, captions = self._get_captions(all_captions)
        return image, caption

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
                 num_workers: int = 1):
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

    def setup(self, stage: str) -> None:
        train_samples, val_samples, test_samples = [], [], []
        with open(self.ann_file, 'r') as f:
            self.data = json.load(f)
            self.ids = list(self.data.keys())
        for k, v in self.data.items():
            captions = v['caption']
            for i in range(min(len(captions), int(self.captions_per_image))):
                caption = captions[i]
                filename = v['img_path']
                augmented_images = v.get('sd_augmentations', [])
                example = Example.fromdict({
                    'id': k,
                    'image': self.img_dir / filename,
                    'augmented_images': [self.img_dir / filename for filename in augmented_images] if self.sd_augmentation else [],
                    'text': caption})
                if v['split'] == 'train':
                    train_samples.append(example)
                elif v['split'] == 'val':
                    val_samples.append(example)
                elif v['split'] == 'test':
                    test_samples.append(example)
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

        if stage == "fit":
            self.train_ds = ArtpediaDataset(train_samples, transform=self.transform
                                            , processor=self.processor, captions_per_image=1
                                            , caption_mode='first', sd_augmentation=self.sd_augmentation)
            self.valid_ds = ArtpediaDataset(val_samples, transform=self.transform
                                            , processor=self.processor, captions_per_image=self.captions_per_image
                                            , caption_mode=self.caption_mode)
        if stage == "validate":
            self.valid_ds = ArtpediaDataset(val_samples, transform=self.transform
                                            , processor=self.processor, captions_per_image=self.captions_per_image
                                            , caption_mode=self.caption_mode)
        if stage == "test":
            self.test_ds = ArtpediaDataset(test_samples, transform=self.transform
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
