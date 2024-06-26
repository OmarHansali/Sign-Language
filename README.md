# Sign Language Recognition with CNN
This project demonstrates real-time [American Sign Language (ASL)](https://en.wikipedia.org/wiki/American_Sign_Language) recognition using a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network).

The goal is to detect ASL hand signs and display the corresponding alphabet on the camera’s display.

## Features
1. Collecting Data:
    - The `Collect_Data.py` script captures images of ASL hand signs using the webcamera.
    - Users can perform different ASL signs to create a diverse dataset.
  
2. Creating and Training the CNN:
    - The `Create-Train-CNN.py` script builds a CNN architecture for ASL recognition.
    - It preprocesses the collected data, splits it into training and validation sets, and trains the model.
    - Model hyperparameters and architecture details are specified in the script.
  
3. Real-Time Testing (4 Labels):
    - The `Test-CNN.py` script captures live video from the webcam.
    - It uses the trained CNN to predict ASL signs in real time.
    - The predicted label (A, B, C or D) is displayed on the camera feed.
  
4. External Model Testing (26 Labels):
    - The `external_model_test.py` script loads a pre-trained external model.
    - It performs real-time ASL recognition on all 26 ASL alphabet labels.

## Installation
1. Clone this repository:
  ```
  git clone https://github.com/OmarHansali/Sign-Language.git
  ```
  ```
  cd Sign-Language
  ```


2. Install the required Python libraries:
  ```
  pip install -r requirements.txt
  ```

## Python Version

This project requires Python 3.10.0.

The current Python version is specified in the `runtime.txt` file.


## Usage
1. Collect Data:
    - Run `collect_data.py` to capture ASL hand signs.
    - Change the saving file direction variable if needed.
    - Press `s` to save the image or `q` to properly stop the program.
    - Organize the collected images into appropriate folders (A, B, …, Z).

  
2. Create and Train the CNN:
    - Adjust hyperparameters in `Create-Train-CNN.py` if needed.
    - Modify the labels variable if needed (Default: `labels = ['A', 'B', 'C', 'D']`).
    - Run `Create-Train-CNN.py` to train the model.
  
3. Real-Time Testing (4 Labels):
    - Execute `Test-CNN.py` to see real-time ASL recognition.
    - Press `q` to properly stop the program.
  
4. External Model Testing (26 Labels):
    - Run `external_model_test.py` with an exported pre-trained model on all the 26 [English alphabet](https://en.wikipedia.org/wiki/English_alphabet).
    - Press `q` to properly stop the program.

## License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/) - see the LICENSE file for details.
