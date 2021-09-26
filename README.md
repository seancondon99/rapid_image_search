# rapid_image_search

This repo contains the code to quickly turn a directory of .TIF images into a set of compressed, information-dense fingerprints, so that one can conduct an image similarity search much more quickly than comparing the original images. Please find more info on the motivations / results of this project on my personal website at this link: https://seancondon99.github.io/ml_sat.html, or keep reading below for the implementation.

## Description of Files

#### ingest_images.py
Loads all the .TIF images in a specified directory and processes them to small image tiles of size stride_len x stride_len. Will save all processed image tiles as numpy arrays in a specified directory. Tiles will now be ready for fingerprint generation.

#### generate_fingerprints.py 
Takes the directories of preprocessed tiles and converts them into information-dense fingerprints by passing them through a pretrained CNN. The code by default will use ResNet-18, and some tinkering will likely be required if you wish to use a different pretained CNN.

#### distance_calc.py
Calculates the distance between a query fingerprint and all other fingerprints in the database, and plots the results. 

## How to Run

### Downloading Dataset

### Tiling Images

### Generating Fingerprints

### Running Query Search

