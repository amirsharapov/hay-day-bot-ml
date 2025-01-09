# Overview

This repository contains data for training a ML model to detect and classify objects in the game Hay Day by Supercell.

## Table of Contents

- [Overview](#overview)
- [Entities](#entities)
  - [Sample](#sample)
  - [Model](#model)
- [Data Structures](#data-structures)
  - [Sample Repository](#sample-repository)
  - [Sample](#sample)
  - [Sample Metadata](#sample-metadata)
  - [Sample Polygons](#sample-polygons)
  - [Sample Augmentation Metadata](#sample-augmentation-metadata)
  - [Sample Augmentation Polygons](#sample-augmentation-polygon)
  - [Model Repository](#model-repository)
  - [Model](#model-metadata)
  - [Model Metadata](#model-metadata)
- [Framework-Specific Data](#framework-specific-data)
  - [Ultralytics](#ultralytics)
  - [Anylabeling](#anylabeling)

# Entities:

Two main entities that are used in this repository: Samples and Models.

## Sample

Samples are individual images used to train the model.
Each sample is stored as a directory with the following naming convention:

`sample_N`

where N is the sample number. Each directory will also contain the following items:

- `sample_N/_meta.json` - The metadata for the image and it's augmentations.
- `sample_N/image.jpg` - The original image file.
- `sample_N/polygons.json` - The list of polygons for the image.
- `sample_N/augmentations/` - The list of augmentations for the image.
- `sample_N/augmentations/augmentation_N/_meta.json` - The metadata for the augmented image.
- `sample_N/augmentations/augmentation_N/image.jpg` - The augmented image file.
- `sample_N/augmentations/augmentation_N/polygons.json` - The list of polygons for the augmented image.

## Model

Models are trained using the samples in this repository.
Each model is stored as a directory with the following naming convention:

`model_N.pt`

where N is the model number. Each directory will also contain the following items:

- `model_N/_meta.json` - The metadata for the model.
- `model_N/best.pt` - The model file.
- `model_N/last.pt` - The model file.
- `model_N/ultralytics_results/` - The results of the model training.

# Data Structures

## Sample Metadata

### JSON File

```
{
    "image": "
}
```

## Persistent Data Format

Every sample contains the following files:

### sample_N.jpg

This contains the original image file.
Compressed to JPEG format because PNG is too large for the dataset and pixel perfect is not required.

### sample_N_meta.json

This contains the metadata for the image.
The metadata should be in JSON format and contain the following fields:

- ``

### sample_N_polygons.json

This contains the list of polygons for the image.

#### Sample:

```
{
    "polygons": [
        {
            "points": [
                [0, 0],
                [1, 1],
                [2, 2]
            ],
            "label": "label",
            "source": "source"  # 'Anylabeling' or specific registered ML model
        }
    ]
}
```

### sample_N.json

