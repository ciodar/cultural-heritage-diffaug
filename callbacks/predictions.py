from lightning import Callback
import wandb


class LogPredictionSamplesCallback(Callback):
    """Log prediction samples to W&B.
    """
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
        if batch_idx == 0:
            wandb_logger = pl_module.logger
            tokenizer = pl_module.tokenizer
            batch, _ = batch
            x, y = batch['pixel_values'].detach().cpu(), batch['labels'].detach().cpu()
            # print max 20 images
            n = max(x.shape[0], 20)

            images = [img for img in x[:n]]
            captions = [f'Prediction: {y_pred} - Ground Truth: {y_i}'
                        for y_pred, y_i in zip(tokenizer.batch_decode(outputs[:n], skip_special_tokens=True),
                                               tokenizer.batch_decode(y[:n], skip_special_tokens=True))]

            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger.log_image(
                key='sample_images',
                images=images,
                caption=captions)