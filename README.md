# song-extender
Choose a song and the model will extend it by using ML tools. 

# Folder Structure

TODO

# Meeting Protocol

## 5th March

- Sadie set up BeatNet and All-in-One: BeatNet works well for extracting the up- and downbeat timestamps, all-in-one is able to detangle background, voice, drums, but not very well, only recognizes beats sometimes, does not really find segments
- could use Sadies simulation code for generating random combinations of a song
- only look at a short period of time of a song excerpt, for example 10 seconds of a 30 seconds excerpt, and compare the mutation of one of the song parts to another
- compare by computing a similarity measure
- for next time: Sadie set up evolution model, Marja set up model that creates a similarity measure

## 10th March 

- Sadie set up Discrete Event Simulator, output is good, but ranges of notes need to be improved
- goal for next meeting: Marja set up CNN for generator (generate input parameters for discrete event simulator) and discriminator (binary classifier for spectrogram of generated midi-file and original spectrogram)


## 13th March

- Marja started implementing the GAN structure according to this website: https://freedium.cfd/https://medium.com/geekculture/deep-convolutional-generative-adversarial-network-using-pytorch-ece1260acc47
- problem: how do we generate enough data to train on from just one song?
- pretrain on bigger data corpus, finetune on song data
- randomly manipulate song data, randomly add data from one song together to create more data, ...
- first only train on songs from one genre to have more data, then think about finetuning
- Sadie: create useful output from simulator for discriminator
- Marja: fix dimensions of GAN, input for simulator: nxn matrix, n=16, 16x20, values from 0-1

## 17th March

- Marja updated the output dimensions of GAN networks
- Sadie implemented midi to audio file converter and made changes to simulator
- Marja: randomly slice the input audio for more input clips of 5 seconds, requirements.txt
- Sadie: connect simulator to GAN 
