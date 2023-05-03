import json

import lightning as L
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pathlib as pl
import random
from transformers import AutoProcessor

# avoids PIL.Image.DecompressionBombError
# https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
Image.MAX_IMAGE_PIXELS = None


class ArtpediaDataset(Dataset):
    def __init__(self, root, split, transform=None, processor=None, captions_per_image=10):
        file = pl.Path(root) / '{}.json'.format(split)
        with open(file) as f:
            j = json.load(f)
            self.data = list(j.values())
        self.split = split
        self.transform = transform
        self.processor = processor
        self.captions_per_image = captions_per_image

    def __getitem__(self, i):
        image_path = self.data[i]['img_url']
        image = Image.open(image_path).convert('RGB')
        # randomly sample one visual sentence
        all_captions = self.data[i]['visual_sentences']

        if self.transform is not None:
            image = self.transform(image)
        # encoding = self.processor(images=image, text=random.sample(labels, 1), padding="max_length", return_tensors="pt")
        # remove batch dimension
        # encoding = {k: v.squeeze() for k, v in encoding.items()}
        if self.split == 'train':
            caption = random.sample(all_captions, 1)
            return image, caption
        else:
            # Make a list of captions of a fixed length for batching purposes
            if len(all_captions) < self.captions_per_image:
                captions = all_captions + [random.choice(all_captions) for _ in
                                           range(self.captions_per_image - len(all_captions))]
            else:
                captions = random.sample(all_captions, k=self.captions_per_image)
            return image, captions

    def __len__(self):
        return len(self.data)

    def _caption_collate(self, batch):
        images, all_captions = zip(*batch)
        captions = [random.choice(c) for c in all_captions]
        encoded = self.processor(images=images, text=captions, padding="max_length",
                                 return_tensors="pt")
        encoded['labels'] = encoded['input_ids']
        if self.split == 'train':
            return encoded
        else:
            return encoded, all_captions


class ArtpediaDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = './data/', batch_size: int = 2, model_name_or_path: str = None
                 , captions_per_image: int = 10, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.captions_per_image = captions_per_image
        # set model name for processor
        self.model_name_or_path = model_name_or_path
        # for now use same batch size for train and test
        self.batch_size = batch_size
        self.num_workers = num_workers
        # TODO: research augmentation for Image Captioning / VQA
        # TODO: Check why Resize crops image
        self.train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(224)
        ])

    def prepare_data(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_ds = ArtpediaDataset(self.data_dir, split='train', transform=self.train_transform
                                            , processor=self.processor, captions_per_image=self.captions_per_image)
            self.valid_ds = ArtpediaDataset(self.data_dir, split='val', transform=self.test_transform
                                            , processor=self.processor, captions_per_image=self.captions_per_image)
        if stage == "test":
            self.test_ds = ArtpediaDataset(self.data_dir, split='test', transform=self.test_transform
                                           , processor=self.processor, captions_per_image=self.captions_per_image)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.train_ds._caption_collate,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, collate_fn=self.valid_ds._caption_collate,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.test_ds._caption_collate,
                          num_workers=self.num_workers)
