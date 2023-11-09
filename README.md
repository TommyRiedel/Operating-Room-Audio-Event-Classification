# Semesterarbeit: "Development of a Deep Learning based Audio Event Classification for the Operating Room"

The time utilization of the operating room can be improved by recognizing the current phase of the operation.
Hardly any work currently uses sounds in the operating room to estimate the current status of the operation, although this is easily accessible and meaningful information.
This semester thesis is therefore attempting to recognize specific sounds in an operating room in order to be able to perform more operations on patients.

# Classes:
  - monopolar coagulation mode
  - bipolar coagulation mode
  - cutting mode
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

<img width="405" alt="Bildschirmfoto 2023-11-09 um 14 11 58" src="https://github.com/TommyRiedel/Operating-Room-Audio-Event-Classification/assets/33426324/5e8b547b-d2b5-4549-b5d7-c79a84051cce">


(mel spectrograms)





# Feature extraction:


# Classification:



Data Augmentation + Transfer Learning
