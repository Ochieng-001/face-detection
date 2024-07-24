# Face Detection and Age/Gender Prediction

## Overview

This project uses OpenCV and deep learning models to detect faces in images and predict their gender and age. The project employs pre-trained models for gender and age classification and showcases how to integrate these with OpenCV's face detection capabilities.

## Features

- **Face Detection**: Detects faces in an image using a Haar cascade classifier.
- **Gender Prediction**: Classifies each detected face as 'Male' or 'Female' using a deep learning model.
- **Age Prediction**: Estimates the age range of each detected face using another deep learning model.
- **Image Handling**: Loads and processes images for face detection and prediction.

## Prerequisites

Ensure you have Python 3.x installed on your system. You will also need the following packages:

- OpenCV
- NumPy

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Ochieng-001/face-detection.git
    cd face-detection
    ```

2. **Create a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download Pre-trained Models**:
    - Ensure the following files are in your project directory:
      - `deploy_gender.prototxt`
      - `gender_net.caffemodel`
      - `deploy_age.prototxt`
      - `age_net.caffemodel`

5. **Run the Script**:
    ```bash
    python face.py
    ```

## Usage

1. **Update the Image Path**: Modify the `image_path` variable in `script.py` to point to your image file.

2. **Run the Detection**: Execute the script to detect faces and predict their gender and age.

3. **View Results**: The detected faces with their predicted gender and age will be printed to the console.



## Contributing

Feel free to fork the repository and submit pull requests. If you encounter any issues or have suggestions for improvements, please open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please reach out to [ochiengpaul654@gmail.com](mailto:ochiengpaul654@gmail.com).

---

> *"The best way to predict the future is to invent it."* - Alan Kay
> 

