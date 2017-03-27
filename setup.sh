#!/bin/bash


echo "Choose size of training data:"
read n
echo -e "Generating the training data (n=$n)...\n"
python ./data_gen.py -n $n
echo -e "Training data is stored in file named ts.\n"

echo -e "Running TensorFlow..."
python ./NNE.py