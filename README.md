# CNN_Polyvore_outfits
This repository contains CNN models trained on the Polyvore dataset for outfit category prediction and compatibility prediction
of 2 outfits.

The Polyvore dataset and other training and testing related files are available in the following link:
https://drive.google.com/uc?id=1yY3k7I14G7-nPnsK8fb9KsqxoX3V92nL&export=download

categorical_model_custom.py and categorical_model_fine_tunes.py contains CNN models to predict outfit category.

compatibility_model_parallel_cnn.py and compatibility_model_siamese.py contains two types of CNN architecture models 
to predict if two outfits are compatible or not.

The model architecture diagrams are available in compatibility_parallel_cnn_architecture.png and 
compatibility_siamese_model_architecture.png files respectively.

data.py, utils.py and data_prep.py are helper files.

In utils.py, make sure to replace Config['root_path'] with the dataset path where the dataset is available after it is
downloaded from the above link.
