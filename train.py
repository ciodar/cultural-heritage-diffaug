import argparse
import collections

from lightning.pytorch.cli import LightningCLI

from data_loader.artpedia import ArtpediaDataModule
from model.model import LitTransformer

def main():
    cli = LightningCLI(LitTransformer, ArtpediaDataModule)


if __name__ == '__main__':
    main()
