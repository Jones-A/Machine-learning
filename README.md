# Machine-learning
By Aquil Jones
 This is a small bit of code meant to recognize the presence of spikes in our    images.
 
 
 ##Data
 Normally our images are both single images and in a 9x9 grid. They are split    into spike and flat classes. I have about 750 images per class to begin with    and an addtional 175 are generated  by adding blur and shuffling around the     locations for a few of the 9x9 images givng 925 per class and a total of 1850   images. Of the 1850 images 200 are used for the validation set
 
 ##Libraries##
 The most used library is the version of keras included as a part of tensorflow  I use the sequential method of building the network as I think it is easier to  follow.I took out dask and a few of the gpu related features from the data      processing for this version as I was told you all do not use that library. I    also use json to serialze and save the model. Besides that openCV is used for   loading and resizing the data as well as a small seperate program I wrote for   generating the
 labels.
 
 ##Performance##
 while I do get around 95% convergence on the training data validation data is   at about 75% accuracy. I think It would have been better to train the 9x9 and   single image versions seperately and then try some sort of transfer learning    technique to get better performance overall.But when I first started this I     did not know how to do that.
 
