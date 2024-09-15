# **Image Feature Extraction and Prediction**

## **Overview**


This project demonstrates how to extract features from images using a pre-trained ResNet50 model and predict numeric values and associated units using a custom neural network model. The goal is to provide accurate predictions based on image inputs.

---

## **Features and Workflow**

### **1. Data Loading and Image Retrieval**

- **Loading Datasets**:
  - **Training Data**: `dataset/train1.csv` contains image URLs and corresponding entity values (numeric values and units).
  - **Test Data**: `dataset/test1.csv` includes image URLs for prediction.

- **Fetching Images**:
  - A function is used to fetch images from URLs.
  - **Image Preprocessing**:
    - **Resizing**: Images are resized to 224x224 pixels to fit the ResNet50 model.
    - **Normalization**: Pixel values are normalized to a range of [0, 1] for consistent input.

---

### **2. Feature Extraction with ResNet50**

- **ResNet50 Initialization**:
  - **Pre-trained Model**: The ResNet50 model is initialized with ImageNet weights.
  - **Top Layer Removal**: The top classification layer is excluded (`include_top=False`) to focus purely on feature extraction.

- **Extracting Image Features**:
  - Each image is passed through the ResNet50 model, and its feature map is extracted and flattened into a one-dimensional array.
  - These features serve as input for the prediction model.

---

### **3. Data Cleaning and Preparation**

- **Entity Value Processing**:
  - A custom function parses the `entity_value` column in the dataset, separating the **numeric values** and their respective **units**.
  - **Data Cleaning**: Invalid or malformed entries are dropped to ensure the model is trained on high-quality data.

- **Unit Label Encoding**:
  - Units are encoded into numeric values using **`LabelEncoder`**, transforming text-based units into integer labels for classification.

- **Feature Alignment**:
  - The feature array is aligned with the cleaned dataset to ensure consistency between features and target values.

---

### **4. Model Definition and Training**

- **Model Architecture**:
  - **Input Layer**: Accepts image features from ResNet50.
  - **Shared Dense Layers**: Two fully connected layers with ReLU activation for feature processing.
  - **Output Layers**:
    - **Numeric Output**: A single neuron for predicting the numeric value (regression task).
    - **Unit Output**: A softmax layer to classify the unit (classification task).

- **Model Compilation**:
  - The model is compiled with two separate loss functions:
    - **Numeric Prediction**: Mean Squared Error (MSE).
    - **Unit Classification**: Sparse Categorical Crossentropy.
  - **Evaluation Metrics**:
    - **Numeric**: Mean Absolute Error (MAE).
    - **Unit**: Classification accuracy.

- **Training the Model**:
  - The model is trained on the training dataset using a **20% validation split** to monitor performance.
  - **Epochs and Batch Size**: The model trains for 10 epochs with a batch size of 32.

---

### **5. Testing and Making Predictions**

- **Feature Extraction for Test Data**:
  - The same feature extraction process is applied to the test images using ResNet50.
  
- **Making Predictions**:
  - The trained model is used to predict:
    - **Numeric Values**: The regression output.
    - **Units**: The classification output.
  - **Formatting the Predictions**: The predicted numeric values and units are combined into a readable format.

- **Generating the Submission File**:
  - The predictions are saved into `submission.csv` with two columns:
    - **Index**: Corresponding to the test data index.
    - **Prediction**: The predicted numeric value and unit.

---

## **File Descriptions**

- **`dataset/train1.csv`**: The training dataset with image URLs, numeric values, and units.
- **`dataset/test1.csv`**: The test dataset with image URLs.
- **`submission.csv`**: The final submission file containing predicted values and units for the test data.

---

## **Usage**

 **Install dependencies**:
  Note : All required dependencied are present in requirements.txt