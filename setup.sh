#!/bin/bash


echo "Choose size of training data:"
read n
echo "Choose size of testing data:"
read t
echo -e "Generating the training data (n=$n), testing data (t=$t)...\n"
let "k=$t+4"
python ./data_gen.py -n $n -t $k
echo -e "Training data is stored in file named ts.\n"

echo -e "Running TensorFlow..."
python ./NNE.py -n $t