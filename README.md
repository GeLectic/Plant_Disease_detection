# Plant Disease Detection using Deep Learning

This repository contains a deep learning-based solution for detecting plant diseases from images. The model is designed to classify plant diseases into 38 different categories using Convolutional Neural Networks (CNN).

## Features
- **Robust Architecture**: Utilizes a deep CNN architecture with multiple convolutional and pooling layers to extract rich features from input images.
- **Batch Normalization**: Included to stabilize and accelerate the training process.
- **Dropout Layers**: Added to reduce overfitting and improve generalization.
- **Multi-class Classification**: Supports classification into 38 plant disease categories.

## Model Architecture
The CNN model is implemented using TensorFlow/Keras and comprises the following layers:

1. **Convolutional Layers**: Extract features with increasing depth from the input images using 32, 64, 128, 256, 512, and 1024 filters.
2. **Batch Normalization**: Ensures stable training by normalizing the output of each convolutional layer.
3. **MaxPooling Layers**: Reduce spatial dimensions while retaining significant features.
4. **Dropout Layers**: Added with a rate of 0.3 to prevent overfitting.
5. **Flatten Layer**: Converts the 2D feature maps into a 1D feature vector for input to the dense layers.
6. **Dense Layers**: Includes fully connected layers with 1500, 1000 neurons, and an output layer with 38 neurons (one for each class).

## Data
The dataset used for this project consists of labeled images of plants with various diseases, split into 38 classes. The images are resized to 128x128x3 before being fed into the model.

## Training
The model was trained using the following parameters:
- Input shape: (128, 128, 3)
- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Epochs: Configurable (10)
- Batch size: Configurable (32)

## Results
The model achieves a training accuracy of **98.28%** and a validation accuracy of **97.31%**, demonstrating high performance in classifying plant diseases. Detailed metrics and performance logs are included in the repository.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/plant-disease-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python run.py
   ```
4. Streamlit app link:
   ```bash
    https://plantdiseasedetection-xyz.streamlit.app/
   ```
5. Data set useed -
    ```bash
    https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
   ```

## Requirements
- Python 3.7+
- TensorFlow 2.18.0
- NumPy
- Matplotlib
- OpenCV

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for suggestions and improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Special thanks to open-source datasets and libraries that made this project possible.

