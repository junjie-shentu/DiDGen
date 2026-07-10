# DiDGen

This repository provides the official PyTorch implementation of **DiDGen**.

The model was originally introduced in our *MICCAI* conference paper, **DiDGen: Diffusion-based Dual-task Synthesis for Dermoscopic Data Generation.**

Our extended paper, **Controllable Synthesis of Dermoscopic Images Using Diffusion Models for Enhanced Computer-Aided Diagnosis and Detection,** has been accepted by *Medical Image Analysis*. This journal paper builds upon our previously published MICCAI conference paper by substantially expanding its scope, incorporating new technical contributions, and providing a more comprehensive evaluation and discussion.


## Installation
### 1. Clone the repository
```
git clone https://github.com/junjie-shentu/DiDGen.git
cd DiDGen
```

### 2. Create a virtual environment (recommended)
```
conda create -n didgen python=3.10
conda activate didgen
```

### 3. Install the dependencies
```
pip install -r requirements.txt
```

## Usage
### 1. Generate detailed descriptions for skin lesion imaages using Llama model
```
bash generate_lesion_mask_pair.sh
```

### 2. Finetune the Stable Diffusion model with region-aware attention loss
```
bash run_finetune_SD.sh
```

### 3. Generate skin lesion images with attention visualization
```
bash generate_image_with_attention_map.sh
```

### 4. Generate lesion-mask pairs using the training-free pipeline
```
bash generate_lesion_mask_pair.sh
```

## Citation
If you find this work helpful, please consider citing the following BibTeX entry:
```
@inproceedings{shentu2025didgen,
  title={DiDGen: Diffusion-Based Dual-Task Synthesis for Dermoscopic Data Generation},
  author={Shentu, Junjie and Watson, Matthew and Al Moubayed, Noura},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={74--84},
  year={2025},
  organization={Springer}
}

@article{SHENTU2026104191,
title = {Controllable synthesis of dermoscopic images using diffusion models for enhanced computer aided diagnosis and detection},
journal = {Medical Image Analysis},
volume = {113},
pages = {104191},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2026.104191},
author = {Junjie Shentu and Matthew Watson and Noura Al Moubayed}
}
```
