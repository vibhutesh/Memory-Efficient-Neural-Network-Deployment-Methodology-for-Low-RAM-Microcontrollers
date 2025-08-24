# Memory-Efficient Neural Network Deployment for Low-RAM Microcontrollers

This repository contains the official implementation and the pre-trained model for the paper: **"Memory-Efficient Neural Network Deployment Methodology for Low-RAM Microcontrollers Using Quantization and Layer-Wise Model Partitioning"**, published in the 2024 IEEE Pune Section International Conference (PuneCon).

Our approach achieves **~97.23%** accuracy on the MNIST dataset by using a lightweight neural network. This repository provides all the necessary code, model, and specific library versions to reproduce our results. The main script demonstrates that the model's high accuracy is maintained even after layer-wise partitioning and weight quantization.

***

## How to Reproduce the Results

To ensure 100% reproducibility and avoid library conflicts, it is **highly recommended** to use a Python virtual environment.

### 1. Clone the Repository

```
git clone [https://github.com/vibhutesh/Memory-Efficient-Neural-Network-Deployment-Methodology-for-Low-RAM-Microcontrollers.git](https://github.com/vibhutesh/Memory-Efficient-Neural-Network-Deployment-Methodology-for-Low-RAM-Microcontrollers.git)
cd Memory-Efficient-Neural-Network-Deployment-Methodology-for-Low-RAM-Microcontrollers
```

### 2. Create and Activate a Virtual Environment

**On macOS/Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Required Libraries

Install the specific library versions required for this project using the `requirements.txt` file.

```
pip install -r requirements.txt
```

### 4. Run the Evaluation Script

Execute the main script. It will automatically unzip the model, download the raw MNIST dataset, perform the HOG and PCA preprocessing on-the-fly, and then run a series of evaluations.

```
python evaluate_model.py
```

### Expected Output ✅

The script will first process the data and then print the evaluation results. The accuracy should remain consistently high across all tests, demonstrating the robustness of the methods presented in the paper.

```
--- 1. EVALUATION (ORIGINAL MODEL) ---
Test accuracy of the original model: 0.972

--- 2. EVALUATION (LAYER-WISE PARTITIONED MODEL) ---
Test accuracy of the cascade model: 0.972

--- 3. EVALUATION (INT16 QUANTIZED MODEL) ---
Test accuracy of the int16 quantized model: 0.972

--- 4. EVALUATION (INT8 QUANTIZED MODEL) ---
Test accuracy of the int8 quantized model: 0.972
```

***

## Citation

If you use this code or model in your research, please cite our paper:

```
V. K. Singh, "Memory-Efficient Neural Network Deployment Methodology for Low-RAM Microcontrollers Using Quantization and Layer-Wise Model Partitioning," 2024 IEEE Pune Section International Conference (PuneCon), Pune, India, 2024, pp. 1-6, doi: 10.1109/PuneCon63413.2024.10895526.
