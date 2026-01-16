# DeepDream

A minimal DeepDream implementation in PyTorch, inspired by Google’s original work, using a pretrained VGG16 network.
This script amplifies internal neural activations via gradient ascent to produce surreal, dream-like images.

## Usage
#### 1. DeepDream from a random image
```shell
uv run python dream.py
```

#### 2. DeepDream from an input image
```shell
python dream.py --image path/to/image.jpg
```

#### 3. Save the output
```shell
python dream.py --image image.jpg --save-image
```

#### 4. Customize the dream
```shell
python dream.py \
  --image image.jpg \
  --layer-index 22 \
  --pyramid-levels 5 \
  --growth-rate 1.6 \
  --steps 15 \
  --learning-rate 0.08
```
#### 5. Class-specific dreaming (ImageNet index)
```shell
python dream.py --image image.jpg --classification-index 130
```

This maximizes the activation of a specific ImageNet class instead of a convolutional layer.

## Features

- DeepDream via feature visualization

-  Multi-scale image pyramid

- Random noise initialization

- User-provided images

- Any intermediate VGG16 feature layer

- Optional class-specific dreaming

- GPU acceleration (CUDA if available)

- Simple CLI interface

## Key Arguments

| Argument                  | Description                       |
|---------------------------|-----------------------------------|
|`--image`                  |   Input image path                |
|`--layer-index`            |	VGG16 feature layer index       |
|`--pyramid-levels`         |	Number of scales                |
|`--growth-rate`            |   Pyramid scale factor            |
|`--steps`                  |   Gradient ascent steps per level |
|`--learning-rate`          |	Gradient ascent scalar          |
|`--classification-index`   |   ImageNet class index            |
|`--save-image`             |	Save output to disk             |