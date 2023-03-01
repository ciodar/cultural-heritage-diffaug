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
    def __init__(self, root, split, image_transform, processor):
        file = pl.Path(root) / '{}.json'.format(split)
        with open(file) as f:
            j = json.load(f)
            self.data = list(j.values())
        self.split = split
        self.image_transform = image_transform
        self.processor = processor

    def __getitem__(self, i):
        image_path = self.data[i]['img_url']
        image = Image.open(image_path).convert('RGB')
        # randomly sample one visual sentence
        labels = self.data[i]['visual_sentences']
        if self.image_transform is not None:
            image = self.image_transform(image)
        encoding = self.processor(images=image, text=random.sample(labels, 1), padding="max_length", return_tensors="pt")
            # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        if self.split == 'test':
            encoding['labels'] = labels
        return encoding

    def __len__(self):
        return len(self.data)


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
