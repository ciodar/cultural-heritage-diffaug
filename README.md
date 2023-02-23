## Folder Structure

```
  cultural-heritage-image2text/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── image_captioning.ipynb - demo of image captioning on Artpedia
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py contains Artpedia Dataset and DataLoader
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── runs/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      ├── download.py - utility to download images from Artpedia json metadata
      └── ...
 ```

## Results

|                      | Dataset  | Rouge F1 score | BLEU score | BERT score |
|----------------------|----------|----------------|------------|------------|
| GIT-base (zero-shot) | Artpedia | 0.179          | 0.004      |            |
| GIT-base (2 epochs)  | Artpedia | 0.27           | 0.006      |            |

## TODOs
- [ ] Train on Artpedia
- [ ] Enable train metrics
- [ ] Support OFA (Tiny-Medium-Base)
- [ ] Evaluate on VQA v2

## References
- [GIT](https://huggingface.co/docs/transformers/main/model_doc/git) Huggingface model
- [Pytorch-template](https://github.com/victoresque/pytorch-template) for project structure

