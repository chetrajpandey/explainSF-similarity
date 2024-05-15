# Repository: Solar Flare Prediction and Explanation

This repository contains code for downloading full-disk HMI images from Helioviewer, prediction models for solar flare events, and functions for calculating similarity measures between explanation maps.

## Folder Structure

- **download_mag**: Contains code to download full-disk HMI images from Helioviewer at a cadence of 12 minutes from 2010-2018. The images are sampled at 1-hour intervals (00:00, 01:00, ..., 23:00).
  
- **flare_labels_with_predictions**: Contains CSV files with model predictions, file timestamps, and actual targets for solar flare events.

## Files

- **calculate_similarity.py**: Calculates similarity measures between explanation maps using all three functions defined in `similarity_measures.py`. It loads maps from `load_explanations.py` module and calculates pairwise similarity between explanations, returning average scores for each model-method pair for both with and without blur cases.

- **example_case_study_hadamard.py**: Provides an example case study of visualization for creating ensemble explanations using the Hadamard product.

- **example_case_study_similarity.py**: Demonstrates visualization for explanations generated from multiple methods and models, with and without using Gaussian blur.

- **generate_explanations.py**: Contains code to generate explanations.

- **load_explanations.py**: Loads explanation module for generating explanations.

- **models.py**: Includes model architecture definitions.

- **similarity_measures.py**: Defines functions for Intersection over Union (IOU), cosine similarity, and root mean square error (RMSE) between explanation maps.

## Usage

Each script can be executed individually to obtain desired results. Make sure to have dependencies installed as specified in the requirements file.

## Contributors

- [Chetraj Pandey](https://github.com/chetrajpandey): Project Lead.
