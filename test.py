import random

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    dataset = getattr(module_data, config['test_data_loader']['type'])(**config['test_data_loader']['args'])
    # build model architecture
    model, processor = config.init_ftn('arch', module_arch)()
    logger.info(model)

    # get function handles of loss and metrics
    if config['loss'] is not None:
        loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    if config.resume is not None:
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset)):
            labels = batch[1]
            inputs = processor(images=batch[0], text=random.sample(labels, 1), padding="max_length",
                               return_tensors="pt")
            # TODO: encode max_length in config
            output = model.generate(pixel_values=inputs.pixel_values.to(device), max_length=50)

            #
            # save sample images, or do something with output here
            #
            decoded_output = processor.batch_decode(output, skip_special_tokens=True)
            # computing loss, metrics on test set
            batch_size = output.shape[0]
            if config['loss'] is not None:
                loss = loss_fn(output, processor(text=batch[1][0], return_tensors="pt"))
                total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(decoded_output, [labels]) * batch_size

    n_samples = len(dataset)
    if config['loss'] is not None:
        log = {'loss': total_loss / n_samples}
    else:
        log = {}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
