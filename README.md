# Digit-Recognizer
It's Neural Network, which takes image as input and predict digit 0-9 in it.


This Neural Network is made from explanation of 3Blue1brown's series Neural Netwrks.

I made 2 versions of this NN, both trained on MNIST's training data set.
First Version contains 13002 parameters, and trained on 40,000 images for 10 epochs.
It able to get ~94.5% accurasy.
This NN takes Image of size 28 * 28 pixel, and it has 2 hidden layer, each containing 16 neurons, and 10 neurons in output layer.


Second Version contains 26506 parameters, and was trained on same 40,000 images, but for 15 epochs and was able to get ~96.5% accurasy.
This NN takes Image of size 28 * 28 pixel, and it has 2 hidden layer, each containing 32 neurons and 10 neurons in output layer.


