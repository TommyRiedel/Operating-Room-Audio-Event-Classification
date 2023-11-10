# Semesterarbeit: "Development of a Deep Learning based Audio Event Classification for the Operating Room"

# Motivation
The time utilization of the operating room can be improved by recognizing the current phase of the operation.
Hardly any work currently uses sounds in the operating room to estimate the current status of the operation, although audible information is easily accessible and can be meaningful.
This semester thesis is therefore attempting to recognize specific sounds in an operating room in order to use the operating room more efficiently and therefore being able to help more patients.

# Classes:
  - Monopolar coagulation mode
  - Bipolar coagulation mode
  - Cutting mode
  - Hämolock fast coagulation mode
  - Table up
  - Table down
  - Table tilt
  - Table forth/back
  - Phone call
  - DaVinci sounds
  - Idle

# Generation and preprocessing of the data:
The audio of several open / laparoscopic and robot-assisted (DaVinci) interventions were recorded at the Klinikum Rechts der Isar (OR 9).
Data augmentation and transfer learning are used to reduce the necessary amount of training data.
The recordings are divided into short snippets using Audacity and a window function generates 0.7 second long samples (tradeoff btw accuracy and real-time capability).
Since transfer learning with models pretrained on ImageNet are used for classification, the audio files are converted to spectrograms (image-like).
To increase the influence of the low frequency range (= human hearing) the mel-scale is used (64 filter bands - tradeoff accuracy and computational resources).
The size of the mel spectrogram images are resized to be equal to the model's input size.

Normalized waveform        |  Mel-power spectrogram
:-------------------------:|:-------------------------:
<img alt="Bildschirmfoto 2023-11-09 um 14 11 58" src="https://github.com/TommyRiedel/Operating-Room-Audio-Event-Classification/assets/33426324/5e8b547b-d2b5-4549-b5d7-c79a84051cce"> | <img alt="Bildschirmfoto 2023-11-09 um 14 21 27" src="https://github.com/TommyRiedel/Operating-Room-Audio-Event-Classification/assets/33426324/e39bc789-b0a1-47e2-94f4-896a98d3a35d">

# Feature extraction and Classification:
The classification models consist of CNNs, whereby the weights of individual filters are adjusted in such a way that different features are extracted regardless of their position in the image.
The first filters recognize lower-level features, which are combined to form higher order ones.
Deeper networks can reveal even more complex relationships.

The higher order features are combined and used to estimated the class via a fully-connected layer.

In transfer Learning the knowledge of a previously learned task is used and applied to the other task when the amount of training data is too small to learn it from scratch.
The tasks have to be similar in order to that the filters can extract usefull information for the classification.
From the keras library different models pretrained on the ImageNet dataset are freely available.

# Models:
  - EfficientNetB4
  - InceptionV3
  - ResNet50
  - VGG19
  - **MobileNet**
  - DenseNet169

# Data Augmentation:
Artificially increases the amount of training data (regularization).
Only used to extend the classes with less data and thus keep a balanced class distribution (only training data).
Time and frequency masking are used as methods to augment the spectrograms.
