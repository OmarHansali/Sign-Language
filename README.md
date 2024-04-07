Real-Time American Sign Language (ASL) Recognition with CNN
This project demonstrates real-time American Sign Language (ASL) recognition using a Convolutional Neural Network (CNN).
The goal is to detect ASL hand signs and display the corresponding alphabet on the camera’s display.

## Features
- Collecting Data:
    The `Collect_Data.py` script captures images of ASL hand signs using the webcam.
    Users can perform different ASL signs to create a diverse dataset.
  
- Creating and Training the CNN:
    The `Create-Train-CNN.py` script builds a CNN architecture for ASL recognition.
    It preprocesses the collected data, splits it into training and validation sets, and trains the model.
    Model hyperparameters and architecture details are specified in the script.
  
- Real-Time Testing (4 Labels):
    The `Test-CNN.py` script captures live video from the webcam.
    It uses the trained CNN to predict ASL signs in real time.
    The top 4 predicted labels are displayed on the camera feed.
  
- External Model Testing (26 Labels):
    The `external_model_test.py` script loads a pre-trained external model.
    It performs real-time ASL recognition on all 26 ASL alphabet labels.

## Installation
1. Clone this repository:
  ```
  git clone https://github.com/OmarHansali/Sign-Language.git
  ```
  ```
  cd ASL-Recognition
  ```

2. Install the required Python libraries:
  ```
  pip install -r requirements.txt
  ```

## Usage
- Collect Data:
    Run collect_data.py to capture ASL hand signs.
    Organize the collected images into appropriate folders (e.g., A, B, …, Z).
  
- Create and Train the CNN:
    Adjust hyperparameters in `Create-Train-CNN.py` if needed.
    Run Create-Train-CNN.py to train the model.
  
- Real-Time Testing (4 Labels):
    Execute `Test-CNN.py` to see real-time ASL recognition.
  
- External Model Testing (26 Labels):
    Run 'external_model_test.py' with the exported pre-trained external model.
## License
This project is licensed under the MIT License - see the LICENSE file for details.
