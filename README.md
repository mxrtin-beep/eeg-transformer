# eeg-transformer


#What is Motor Imagery?

Motor Imagery is one of the most powerful inputs in the world of Brain-Computer Interfaces (BCI). Simply put, Motor Imagery is thinking of moving a certain limb, usually the left arm, right arm, legs, and maybe tongue. This represents a very natural input to a BCI device; however, the classification is not nearly as simple as other inputs such as P300 and SSVEP.
Transformers represent an exciting opportunity to improve classification over current methods. I attempted to emulate and modify this paper's strategy, which first employs convolutional layers to reduce the signal into a few elements, and then utilizes self-attention and temporal convolution to produce predictions. This paper uses TensorFlow. Here, I try to paraphrase the strategy into PyTorch and make it more customizable.
