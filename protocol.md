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
