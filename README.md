# IMAGE-CAPTION-GENERATOR

This project implements an Image Caption Generator using deep learning techniques. Given an input image, the model generates a descriptive caption that describes the content of the image.

## Table of Contents
- Overview
- Key Features
- Dataset
- requires dependencies
- Model Evaluation
- Acknowledgments
- Contributions
- Contact

## Overview

Image Caption Generator with CNN & LSTM is a Deep learning project. Image caption generation is a system that comprehends natural language processing & computer vision standards to recognize the connection of the image in English. The project goal is to come up with a caption for an image. “Image captioning is the process of generating descriptions about what is going on in the image”. Our project model will take an image as input and generate an English sentence as output. The project involves training a convolutional neural network (CNN) to extract features from images, and then using a recurrent neural network (RNN) with a long short-term memory (LSTM) architecture to generate captions based on those features. The model is trained using a large dataset of image-caption pairs. For the image caption generator, we will be using the Flickr_8K dataset. The advantage of a huge dataset is that we can build better models. image caption generators have a wide range of applications and can help improve efficiency, accessibility, and accuracy in various industries including: Social media, Education, Advertising, News and media, Healthcare, Entertainment, Security.

Keywords: Deep learning, Image caption, Convolutional Neural network (CNN), Recurrent neural network (RNN), Long short term memory(LSTM), Computer vision, natural language processing.     


## Key Features

- Automatic image caption generation for a given image
- Utilizes pre-trained VGG16 model for image feature extraction
- LSTM-based RNN for text generation
- BLEU score evaluation for caption quality assessment


## Dataset

The Image Caption Generator project uses the Flickr8k dataset, which contains a diverse set of images along with corresponding captions. You can download the dataset from [Flickr8k Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) and [Flickr8k Text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip). Extract the downloaded files and place them in the `datasets` directory.

## requires dependencies:
- os
- pickle
- tqdm
- TensorFlow
- Keras
- NumPy
- PIL
- Matplotlib
- NLTK (for BLEU score calculation)
- streamlit

- Install the required packages 

## Model Evaluation

The quality of the generated captions can be assessed using the BLEU score, which measures the similarity between the generated caption and the ground truth captions. The BLEU score is calculated using the `nltk.translate.bleu_score.corpus_bleu` function. The evaluation is performed on a separate test set.


## Acknowledgments

- Flickr8k Dataset by James J. McAuley is acknowledged for providing the dataset used in this project.

## Contributions

Contributions to the project are welcome! Feel free to open issues or submit pull requests.

## Contact

If you have any questions or suggestions, please feel free to contact with me:

- sham johari - shamjohari101@gmail.com
