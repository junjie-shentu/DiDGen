# DiDGen: Diffusion-based Dual-task Synthesis for Dermoscopic Data Generation

### This is the official PyTorch implementation for the paper DiDGen: Diffusion-based Dual-task Synthesis for Dermoscopic Data Generation.

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
