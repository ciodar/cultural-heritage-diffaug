# Image captioning in Cultural Heritage domain
Project work for Computer Vision exam. Evaluation of image captioning and visual question answering techniques on cultural heritage datasets 

[//]: # (and the impact of an augmentation approach based on diffusion models)

## Project Structure

```
  cultural-heritage-image2text/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── notebooks/ - collection of notebooks for exploration and demonstration of features
  │   ├── image-captioning.ipynb
  │   └── ...
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── artpedia.py contains Artpedia Dataset and DataModule
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models and metrics
  │   ├── model.py - LightningModule wrapper for image captioning
  │   └── metric.py - custom metrics for causal language modeling
  │
  ├── runs/
  │   ├── cultural-heritage/ - trained models are saved here
  │   └── wandb/ - local logdir for wandb and logging output
  │
  └── utils/
      ├── utils.py - small utility functions for training
      └── download.py - utility to download images from Artpedia json metadata
 ```

## Results

### Artpedia
| Model    | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE | BERTScore |
|----------|--------|--------|--------|--------|--------|---------|-------|-------|-----------|
| [GIT](https://huggingface.co/docs/transformers/model_doc/git)  |
| [OFA](https://github.com/OFA-Sys/OFA)  |
| [BLIP](https://huggingface.co/docs/transformers/model_doc/blip) |

## TODOs
- [ ] Train on Artpedia
- [ ] Add BLEU, METEOR, ROUGE-L, CIDEr, SPICE
- [ ] Support OFA (Tiny-Medium-Base)
- [ ] Evaluate on other art datasets

## References
- [Artpedia](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=35) dataset - [paper](https://iris.unimore.it/retrieve/handle/11380/1178736/224456/paper.pdf)

