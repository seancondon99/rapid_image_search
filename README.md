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

### Downloading the Dataset

We first need a large dataset of .TIF images to process into fingerprints. I originally used a dataset of drone images taken from this public repo: https://github.com/dronedeploy/dd-ml-segmentation-benchmark.

You can find more information on this data in the repo above, but essentially we have ~60 very high-resolution drone images taken of cities, natural landscapes, suburbs, etc. The dataset in .TIF form is 10 GB, but when we convert it to numpy arrays of 32-bit floats, its total size is over 70 GB. 

The download link for the drone images can be found in libs/dataset.py, or you can just click this link to download it from google drive. : 'https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0'. Once you have the images, please move them to the rapid_image_search directory inside a folder called 'TIF_files'

### Generating tiles

The next step is to convert these .TIF files into small image tiles, each of size 256 x 256 tiles. First we create a directory inside of rapid_image_search called 'ingested_images', and then we run 'python3 ingest_images.py'

This step will probably take a while, but should print progress updates as it goes. 

### Generating fingerprints

Next we convert all the image tiles into fingerprints by feeding each tile through a pretrained ResNet-18 architecture. To do this, we will first create a directory called 'fingerprints' inside of rapid_image_search. We can then call 'python3 generate_fingerprints.py'

This step will also take a long time, but will create and save dictionaries mapping each image tile to its fingerprint, also printing progress updates along the way.

### K-nearest tiles

Now that we have all our fingerprints processed, we are ready to calculate the k-most-similar image tiles to a query tile. To do this, we just need to run 'python3 distance_calc.py'

By default, the main function of this file will find the 10 most similar images to a random query which is selected uniformly from all possible tiles. A plot of the query and the results will be shown. Note that, if you have not already aggregated all of the .json files in ./fingerprints, you might have to add the line 'save_global_dictionary()' to the main function of distance_calc.py, before the manage_query function is called. This will combine all the data in ./fingerprints into one .json file, and save it in the working directory.


