from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from data_loader.artpedia import ArtpediaDataModule
from model.model import LitTransformer
import wandb


# main.py
class WandbLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--resume_run_id", default="", type=str, help="W&B run ID to resume from"
        )
        parser.link_arguments("resume_run_id", "trainer.logger.init_args.resume",
                              compute_fn=lambda x: "must" if x else "never")

    def before_instantiate_classes(self):
        subcommand = self.config.subcommand
        c = self.config[subcommand]
        run_id = None

        if c.resume_run_id:
            run_id = c.resume_run_id
            api = wandb.Api()
            artifact = api.artifact(f'{c.trainer.logger.init_args.project}/{run_id}', type="model")
            artifact_dir = artifact.download()
            c.ckpt_path = str(Path(artifact_dir) / "model.ckpt")
        else:
            run_id = wandb.util.generate_id()

        # also make sure that ModelCheckpoints go to the right place
        c.trainer.logger.init_args.id = run_id
        for callback in c.trainer.callbacks:
            if callback.class_path == "lightning.pytorch.callbacks.ModelCheckpoint":
                callback.init_args.dirpath = f"checkpoints/{run_id}"


def main():
    cli = WandbLightningCLI(LitTransformer, ArtpediaDataModule, save_config_callback=None)


if __name__ == '__main__':
    main()
