# Semesterarbeit: "Development of a Deep Learning based Audio Event Classification for the Operating Room"
### Learning
  - Tensorflow
  - CNNs (Deep Learning image classification)
  - Data Augmentation (images/spectrograms + audio files)
  - Keras (Transfer Learning)
  - t-SNE

## Motivation
The operating room's estimated time utilization can be dynamically updated by identifying its current phase. 
Despite the accessibility and meaningful nature of audible information, the use of sound in operating rooms for real-time operation status estimation is underutilized.
This semester thesis aims to address this gap by developing a system that identifies specific sounds in the operating room.
The goal is to enhance efficiency, ultimately enabling the medical team to assist more patients effectively.

## Generation and preprocessing of the data (**Preprocessing.py**):
Audio recordings from 23 surgical interventions, including open, laparoscopic, and robot-assisted (DaVinci) procedures, totaling 74 hours and 27 minutes, were captured at Klinikum Rechts der Isar, specifically in operating room 9.
Due to data limitations, certain sounds, such as the surgeon's initial "Schnitt" command or the stapler noise occurring towards the end of procedures like sigmoid resection, couldn't be included.
There is significant variability in the data across the mentioned classes. 
To address data scarcity, transfer learning is employed to minimize the required training data, while data augmentation is utilized to amplify less common sounds.
The recordings are segmented into 0.7-second snippets using Audacity and a window function, striking a balance between accuracy and real-time feasibility.
Since transfer learning involves models pre-trained on ImageNet, the audio files are converted into spectrograms resembling images.
To enhance the impact of the low-frequency range (within human hearing), the mel-scale with 64 filter bands is applied, considering the tradeoff between accuracy and computational resources.
Additionally, the size of the mel-power spectrogram images is adjusted to align with the input size requirements of the corresponding model.

Normalized waveform        |  Mel-power spectrogram
:-------------------------:|:-------------------------:
<img alt="Bildschirmfoto 2023-11-09 um 14 11 58" src="https://github.com/TommyRiedel/Operating-Room-Audio-Event-Classification/assets/33426324/5e8b547b-d2b5-4549-b5d7-c79a84051cce"> | <img alt="Bildschirmfoto 2023-11-09 um 14 21 27" src="https://github.com/TommyRiedel/Operating-Room-Audio-Event-Classification/assets/33426324/e39bc789-b0a1-47e2-94f4-896a98d3a35d">

### Classes:
In an initial study with a smaller data set (200 spectrograms per class), all of the following classes were considered.
However, due to the limited data availability, the classes marked with an asterisk (*) had to be excluded from further investigations with larger datasets (1000 spectrograms per class).

  - Monopolar coagulation mode
  - Bipolar coagulation mode
  - Cutting mode (*)
  - Hämolock fast coagulation mode
  - Table up
  - Table down
  - Table tilt
  - Table forth/back
  - Phone call (*)
  - DaVinci sounds
  - Idle

## Feature extraction and Classification (**model_final.py**):
Both feature extraction and classification are performed using the following deep learning models pretrained on ImageNet.
These models comprise multiple convolutional layers designed to extract meaningful features, irrespective of their spatial position in the image/spectrogram.
In the initial layers, low-level features such as curves and edges are extracted, progressively abstracted to higher-order features in the deeper layers.
The objective is to transfer the knowledge gained from image-based tasks, specifically the weights of the filters, to the spectrogram domain.
To achieve this, the weights of the first layers are kept frozen, and only the final layers responsible for classification are retrained
Ensuring the similarity between the data from the two tasks is crucial to guarantee that the extracted features remain meaningful for the new classification task.
Various pre-trained models from the **keras** library, originating from the ImageNet dataset, are freely accessible for this purpose.
This approach proves especially valuable when dealing with limited data, as it leverages the pre-existing knowledge encoded in the pre-trained models, circumventing the need to train a deep model from scratch, which is often impractical with small datasets.

### Models:
  - EfficientNetB4
  - InceptionV3
  - ResNet50
  - VGG19
  - **MobileNet**
  - DenseNet169

## Data Augmentation (**Preprocessing.py**):
This study employs augmentation techniques to artificially enhance the training data, particularly focusing on classes with limited available data. 
Two distinct augmentation approaches are investigated in this work.
Firstly, the spectrograms undergo augmentation through the process of masking out certain time and/or frequency channels. 
This method is implemented using the script **Augment_spec.py**.
Alternatively, the original wave-files are augmented using two techniques: amplification and the addition of white noise. 
This augmentation is carried out using the script **Augment_wave.py**.
By comparing these two augmentation methods, the study aims to evaluate their effectiveness in improving model robustness and performance, especially when dealing with classes that have a scarcity of training data.

Augmented spectrogram       |  Augmented wave-file
:-------------------------:|:-------------------------:
<img width="500" alt="Bildschirmfoto 2023-11-13 um 13 04 11" src="https://github.com/TommyRiedel/Operating-Room-Audio-Event-Classification/assets/33426324/a1fe10f8-4459-4d17-9cc6-31004c258358"> | <img width="450" alt="Bildschirmfoto 2023-11-13 um 13 05 28" src="https://github.com/TommyRiedel/Operating-Room-Audio-Event-Classification/assets/33426324/f085fe41-6d48-46b9-9f28-74e515a7dce6">

## Results:
The initial study, conducted on the smaller dataset, reveals that the pre-trained MobileNet model yields the most favourable results. 
The advantage over the DenseNet169 model is the shorter time required per epoch
Interestingly, the accuracies for the MobileNet model, trained with various augmentation strategies, exhibit minimal differences.
Although the augmented data is (supposed to be) used only for training the algorithm, the classification performance on augmented spectrograms is poor, suggesting that they are too different from the normal data and therefore do not provide any added value for training the network.

### t-SNE:
t-SNE, a non-linear dimensionality reduction technique, is employed in this study to explore whether the MobileNet model can effectively extract meaningful features for distinguishing between different sounds.
By leveraging t-SNE, the goal is to visualize and analyze the arrangement of data points in a reduced-dimensional space, with an expectation that similar sounds will be positioned in close proximity to each other.
This analysis provides insights into the model's capacity to discern distinct acoustic features and highlights its ability to capture meaningful representations within the data.