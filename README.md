### Update 10-April-2017
And now it works with Python3 and Tensorflow 1.1.0

### Update 02-Jan-2017
I updated this repo. Now it works with Tensorflow 0.12. In this readme I comment on some new benchmarks

### LSTM for time-series classification
This post implements a Long Short-term memory for time series classification(LSTM). An LSTM is the extension of the classical Recurrent Neural Network. It has more flexibility and interpretable features such as a memory it can read, write and forget.

## Aim
This repo aims to show the minimal Tensorflow code for proper time series classification. The main function loads the data and iterates over training steps. The *tsc_model.py* scripts contains the actual model.
This repo contrasts with [another project](http://robromijnders.github.io/CNN_tsc/) where I implement a similar script using convolutional neural networks as the model

## Data and results
The code generalizes for any of the [UCR time series](http://www.cs.ucr.edu/~eamonn/time_series_data/). With the parameter *dataset* you can run the code on any of their datasets.
For your interests, you may compare performances with the nice overview in [this paper.](https://arxiv.org/pdf/1603.06995v4.pdf) They benchmark their CNN and other models on many of the UCR time series datasets
This code works amongst others for
  * __Two_Patterns__ where it achieves state-of-the-art, bein 100% test accuracy
  * __ChlorineConcentration__ where it achieves state-of-the-art, being 80% test accuracy

# Credits
Credits for this project go to [Tensorflow](https://www.tensorflow.org/versions/r0.7/tutorials/recurrent/index.html#recurrent-neural-networks) for providing a strong example, the [UCR archive](http://www.cs.ucr.edu/~eamonn/time_series_data/) for the dataset and my friend Ryan for strong feedback.

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com
