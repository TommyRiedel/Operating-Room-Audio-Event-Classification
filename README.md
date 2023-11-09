# Semesterarbeit: "Development of a Deep Learning based Audio Event Classification for the Operating Room"

The time utilization of the operating room can be improved by recognizing the current phase of the operation.
Hardly any work currently uses sounds in the operating room to estimate the current status of the operation, although this is easily accessible and meaningful information.
This semester thesis is therefore attempting to recognize specific sounds in an operating room in order to be able to perform more operations on patients.

# Generation and preprocessing of the data:
The audio of several open / laparoscopic and robot-assisted (DaVinci) interventions were recorded at the Klinikum Rechts der Isar (OR 9).
Data augmentation and transfer learning are used to reduce the necessary amount of training data
(mel spectrograms)

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


Feature extraction:


Classification:



Data Augmentation + Transfer Learning
