from lightning.pytorch.cli import LightningCLI

from data_loader.artpedia import ArtpediaDataModule
from model.model import LitTransformer

PL_GLOBAL_SEED = 42

def main():
    cli = LightningCLI(LitTransformer, ArtpediaDataModule, save_config_callback=None)


if __name__ == '__main__':
    main()
