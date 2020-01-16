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
cd into the src folder and from the command line type 

"python transformer.py string train_length"

where string is a contiguous string of english alphabets
and train_length is how many characters you want to train the model on

# Output
The transformer.py script outputs a file 'points.csv'
To visulaize the points run:
"python visualization.py"

Note: matplotlib must be installed for visualization.py to run
