# Transformer
Transformer for handwriting synthesis

# Overview
The project contains a single python script to generate 2-D points corresponding to a contiguous english text.

# Requirements

Tensorflow: 2.0.0
Keras: 2.3.1
pandas
numpy

# Running the script
cd into the src folder and from the command line execute these commands in order
> **wget https://polybox.ethz.ch/index.php/s/uqZq0nPDW6BcKyF/download**

> **mv download train_x.csv**

> **wget https://polybox.ethz.ch/index.php/s/YemP53P4IYgnvZJ/download**

> **mv download train_y.csv**

> **python transformer.py string train_length**

where *string* is a contiguous string of english alphabets
and *train_length* is how many characters you want to train the model on

# Output
The transformer.py script outputs a file 'points.csv'
To visualize the points, run:
> **python visualization.py**

Note: matplotlib must be installed for visualization.py to run
