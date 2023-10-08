# eeg-transformer


# What is Motor Imagery?

Motor imagery is one of the most powerful inputs in the world of Brain-Computer Interfaces (BCI). Motor imagery involves thinking of moving a certain limb, usually a hand, a leg, or the tongue. Motor imagery is unique in that the input to the machine is within the user's direct control via their imagination. Other BCI methods, such as P300 and SSVEP, rely on the user's neural response to a stimulus shown on a screen. However, unlike P300 and SSVEP, which produce clear EEG signals, motor imagery's affect on EEG is poorly understood. Machine learning is the go-to method for classifying motor imagery data.
Transformers represent an exciting opportunity to improve classification over current methods. I attempted to emulate and modify this paper's (https://github.com/Altaheri/EEG-ATCNet/tree/main) strategy, which first employs convolutional layers to reduce the signal into a few elements, and then utilizes self-attention and temporal convolution to produce predictions. This paper uses TensorFlow. Here, I try to paraphrase the strategy into PyTorch and make it more customizable. The researchers achieve a subject-independent accuracy of 71%, which I try to replicate here.

# The Model

The paper provides this diagram for their model.

![alt text](https://mxrtin-beep.github.io/images/atcnet.png)

This diagram shows specific numbers for the dimensions of various layers. I started by hard-coding these numbers to make a working model and later made them variable. This code can be found in model.py.

# Data Processing

I used the same data as the paper, the BCI-IV-2a competition data, found here: https://www.bbci.de/competition/iv/#dataset2a. It has four classes: left hand, right hand, legs, and tongue. In processing.py, I load the data and using the MNE package, extract the actual trials and their labels into training and testing data.
I attempted a few preprocessing strategies to attempt to increase model accuracy.

* Bandpass Filter: Little to no change.
* Normalizing: Possible slight improvement.
* Augmentation: Much worse results.


I plan to try utilizing power spectral data instead of raw EEG data. This would add another dimension, going from (Batch, Depth, N_channels, Time) to (Batch, Depth, N_channels, Frequency, Time).

# Training and Testing

From the paper: The proposed model outperforms the current state-of-the-art techniques in the BCI Competition IV-2a dataset with an accuracy of 85.38% and 70.97% for the subject-dependent and subject-independent modes, respectively. The subject-dependent approach tested the model on the same subject that it trained on. The subject-independent approach trained the model on all subjects but one and tested it on the excluded subject. I have found that the latter accuracy highly depends on which is the excluded subject.

I would achieve a training accuracy of 80-85% and a validation accuracy of 60-65% with normalization. Here is a typical set of results. The classes are encoded as follows:
* 0: Left
* 1: Right
* 2: Foot
* 3: Tongue

![alt text](https://mxrtin-beep.github.io/images/2_accuracies_10000.png)
![alt text](https://mxrtin-beep.github.io/images/2_10000_confusionmatrix.png)

# Conclusion

This project has not concluded yet. However, I could never reliably achieve quite the 70% accuracy of the subject-independent model. My model would hover around 65%.
Most of the parameters I changed did not have a large effect on the results. In the future, I would like to try to use power spectral data, as well as augmentation. My augmentation strategy involved shifting the a recording forward or backward by 50 time points to increase the dataset. I would like to understand why I failed and use augmentation to improve my training.
