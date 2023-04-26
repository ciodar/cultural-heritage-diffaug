import json

from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
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
        if self.split == 'train':
            return encoded
        else:
            return encoded, all_captions


class ArtpediaDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, shuffle=True, validation_split=0.0, num_workers=1,
                 processor=None):
        transform = transforms.Compose([
            transforms.Resize((224, 224))
        ])
        processor = AutoProcessor.from_pretrained(processor)
        self.data_dir = data_dir
        self.dataset = ArtpediaDataset(data_dir, split, image_transform=transform, processor=processor)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
