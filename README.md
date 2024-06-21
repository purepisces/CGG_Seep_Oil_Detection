# CGG Seep Oil Seep Detection Exercise

This exercise involves image segmentation for oil seep detection using synthetic aperture radar (SAR) images. The images are 256x256 pixels, and each pixel is classified as either non-seep (0) or one of seven classes of seeps (1-7). The objective is to segment regions containing seeps and optionally classify the seeps.

## How to Run the Code

To run this code, you need to upload the `seep_detection.tar.gz` file to Google Colab. The code is designed to utilize GPU for faster processing. Follow these steps:

1. Open Google Colab and create a new notebook.
2. Upload the `seep_detection.tar.gz` file to the Colab environment.
3. Ensure the notebook is set to use GPU by navigating to `Runtime > Change runtime type > GPU`.
4. Run the full code provided below in the Colab notebook.

## Dataset and DataLoader Setup

In this section, we set up the dataset and data loaders for the seep detection task using PyTorch. This involves defining a custom `DataFolder` class to handle loading and preprocessing of the images and masks, and creating data loaders for training, validation, and evaluation.

## UNet Model Definition

We define a UNet model for the seep detection task. The UNet architecture is commonly used for image segmentation tasks due to its ability to capture both low-level and high-level features through its encoder-decoder structure.

### UNet Architecture

1. **Encoder**: The encoder consists of four convolutional blocks that progressively downsample the input image and capture its features at different scales.
   - `conv_block` is used to define each convolutional block.
   - Layers: `encoder1` (64 channels), `encoder2` (128 channels), `encoder3` (256 channels), `encoder4` (512 channels).
2. **Bottleneck**: The bottleneck layer connects the encoder and decoder, providing a bridge where the most compressed feature representation of the input image is obtained.
   - Layers: `bottleneck` (1024 channels).
3. **Decoder**: The decoder consists of four upsampling blocks that progressively reconstruct the image from the bottleneck features, combining them with corresponding features from the encoder through skip connections.
   - Layers: `upconv4`, `decoder4` (512 channels), `upconv3`, `decoder3` (256 channels), `upconv2`, `decoder2` (128 channels), `upconv1`, `decoder1` (64 channels).
4. **Final Convolution**: A final 1x1 convolution layer maps the output to the desired number of classes.
   - Layer: `final_conv` (8 classes).

## Early Stopping

An `EarlyStopping` class is defined to monitor the validation loss and stop training when the model stops improving. This helps prevent overfitting and ensures the model is saved at its best state.

## Training and Validation Functions

- `train_one_epoch`: Trains the model for one epoch and calculates the average training loss.
- `validate_one_epoch`: Validates the model for one epoch and calculates the average validation loss.

## Model Training

The `train_model` function handles the entire training process, including calling the training and validation functions for each epoch, checking the early stopping criterion, and saving the best model.

## Model Evaluation

The `evaluate_model` function evaluates the model on the evaluation dataset and calculates the average evaluation loss.

## Prediction and Visualization

- `save_predictions`: Saves the prediction masks to the specified output directory.
- `visualize_prediction`: Visualizes the input image, ground truth mask, and prediction mask.
- `predict_and_save`: Generates predictions for the entire dataset, calculates the prediction loss, saves the predictions, and visualizes them.

### Hyperparameters

- `epochs`: 300
- `lr`: 0.001
- `patience`: 10
- `min_delta`: 0.001

### Example Usage

The code demonstrates how to create an instance of the UNet model, set up the optimizer and loss function, and call the `train_model` and `predict_and_save` functions to train, evaluate, and generate predictions.

## Future Improvements

1. **Handling Class Imbalance**: The masks in the dataset are class imbalanced, meaning some classes are underrepresented compared to others. This can affect the model's ability to learn effectively for all classes. To address this, we could use the Dice coefficient or a weighted loss function for cross-entropy, which gives more importance to the underrepresented classes and improves overall performance.


2. **Learning Rate Scheduling**: Implementing learning rate scheduling can help the model converge better. By reducing the learning rate when the validation loss plateaus, we can ensure the model continues to make small, incremental improvements rather than oscillating or getting stuck in suboptimal states.


3. **Metrics**: In addition to loss, using metrics such as Intersection over Union (IoU) or Dice coefficient would provide a better understanding of the model's segmentation performance. These metrics are particularly useful for segmentation tasks as they measure the overlap between the predicted and ground truth masks, giving a more comprehensive view of how well the model is performing.
