# ANN Practice - Customer ExpectedSalary Prediction Regression
A comprehensive Artificial Neural Network project for predicting customer churn using deep learning.

## 🎯 Project Overview

This project implements an Artificial Neural Network (ANN) to predict whether a customer will leave a bank. It includes model training, evaluation, and deployment with an interactive Streamlit web application.

## 📦 Tech Stack

### Core Libraries
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Latest-D00000?logo=keras&logoColor=white)](https://keras.io/)

### Data Processing & Analysis
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

### Visualization & Deployment
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-2.13+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)

## 📁 Project Structure

```
ANN/
├── app.py                    # Streamlit web application
├── model.h5                  # Pre-trained ANN model
├── Churn_Modelling.csv       # Dataset
├── requirements.txt          # Python dependencies
├── experiments.ipynb         # Experimental notebook
├── prediction.ipynb          # Prediction notebook
├── regression.ipynb          # Regression analysis
├── logs/                     # TensorBoard logs
│   ├── fit20260214-152207/
│   └── fit20260214-193201/
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Conda or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/valiantProgrammer/ANN-Practice.git
   cd ANN-Practice
   ```

2. **Create and activate virtual environment**
   ```bash
   conda create -n ann-env python=3.11
   conda activate ann-env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The project uses the **Churn_Modelling.csv** dataset containing customer information and churn status. The model predicts whether a customer will exit the bank based on various features.

## 🤖 Model Architecture

The ANN model includes:
- **Input Layer**: Features from preprocessing
- **Hidden Layers**: Multiple dense layers with ReLU activation
- **Output Layer**: Sigmoid activation for binary classification
- **Optimization**: Adam optimizer with binary cross-entropy loss

### Training
- **Batch Size**: 32
- **Epochs**: Configured for optimal convergence
- **Validation Split**: 20%
- **Monitoring**: TensorBoard integration for real-time tracking

## 💻 Usage

### Using the Web Application

```bash
streamlit run app.py
```

The web app provides:
- Model predictions for new customers
- Interactive interface for data input
- Real-time probability calculations
- Performance metrics visualization

### Using Jupyter Notebooks

**For experiments:**
**For regression analysis:**
```bash
jupyter notebook regression.ipynb
```

## 📈 Training Results & TensorBoard Monitoring

### Training Metrics

The model has been trained with comprehensive logging of training and validation metrics:

**Tracked Metrics:**
- **Loss**: Binary cross-entropy loss for training and validation
- **Accuracy**: Classification accuracy on training and validation sets
- **Epochs**: Multiple training runs logged for comparison

### View Training Graphs

To visualize the training loss and accuracy graphs interactively, run TensorBoard:

```bash
tensorboard --logdir=logs/
```

Then open **`http://localhost:6006`** in your browser.

**Available Graphs in TensorBoard:**
- 📊 **Training Loss**: Shows how the model's loss decreased over epochs
- 📊 **Validation Loss**: Indicates generalization performance
- 📊 **Training Accuracy**: Monitors training set accuracy improvement
- 📊 **Validation Accuracy**: Validates model performance on unseen data

### Training Runs

The project includes multiple training runs logged in separate directories:

| Run | Directory | Purpose |
|-----|-----------|---------|
| Run 1 | `logs/fit20260214-152207/` | Initial model training |
| Run 2 | `logs/fit20260214-193201/` | Model refinement & tuning |

You can compare multiple training runs in TensorBoard by loading all logs at once:

```bash
tensorboard --logdir=logs/ --reload_interval=10
```

This will display scalars for both training runs, allowing you to compare their performance side by side.

## 📋 Requirements

All dependencies are listed in `requirements.txt`:

```
tensorflow>=2.13.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorboard>=2.13.0
matplotlib>=3.7.0
streamlit>=1.32.0
```

## 🔧 Troubleshooting

### Protobuf Compatibility Issues
If you encounter protobuf conflicts, ensure you're using compatible versions:
- TensorFlow requires protobuf>=5.28.0
- Streamlit 1.32.0+ supports protobuf 5.x

### Model Loading
The pre-trained model is saved as `model.h5`. To load it:

```python
from tensorflow.keras.models import load_model
model = load_model('model.h5')

```

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Guide](https://keras.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Guide](https://scikit-learn.org/stable/)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**[valiantProgrammer](https://github.com/valiantProgrammer/)**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Last Updated**: February 15, 2026
