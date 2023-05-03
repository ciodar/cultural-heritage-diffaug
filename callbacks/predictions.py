from lightning import Callback
import wandb


class LogPredictionSamplesCallback(Callback):

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            wandb_logger = pl_module.logger
            tokenizer = pl_module.tokenizer

            batch, all_labels = batch
            n = 2
            x, y = batch['pixel_values'].detach().cpu(), batch['labels'].detach().cpu()

            images = [img for img in x[:n]]
            captions = [f'Prediction: {y_pred} - Ground Truth: {y_i}'
                        for y_pred,y_i in zip(tokenizer.batch_decode(outputs[:n],skip_special_tokens=True),tokenizer.batch_decode(y[:n],skip_special_tokens=True))]

            # Option 2: log images and predictions as a W&B Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]

            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger.log_image(
                key='sample_images',
                images=images,
                caption=captions)
            # log table to wandb
            # wandb_logger.log_table(
            #     key='sample_table',
            #     columns=columns,
            #     data=data,
            #     step=trainer.global_step)