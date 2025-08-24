# Suppress TensorFlow and other warnings for a cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # FATAL
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import uuid
import zipfile
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from skimage.feature import hog

# ==============================================================================
# 0. SETUP MODEL PATHS AND AUTO-UNZIP
# ==============================================================================
MODEL_DIR = 'model'
MODEL_ZIP_PATH = os.path.join(MODEL_DIR, 'MODEL_16PCA_16ELU_12ELU_12ELU_97.23_ppc_4_4_cpb_3_3_o_10_new.tf.zip')
MODEL_EXTRACT_PATH = os.path.join(MODEL_DIR, 'Actual_16PCA_16ELU_12ELU_12ELU_97.23_ppc_4_4_cpb_3_3_o_10_new.tf')

# Check if the model directory exists, if not, unzip the model file
if not os.path.exists(MODEL_EXTRACT_PATH):
    print("Model directory not found. Unzipping model...")
    try:
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        print("Model unzipped successfully.")
    except FileNotFoundError:
        print(f"Error: Zipped model not found at '{MODEL_ZIP_PATH}'.")
        print("Please ensure the model zip file is in the 'model' directory.")
        exit()
else:
    print("Model directory already exists. Skipping unzipping.")

# ==============================================================================
# 1. DEFINE CUSTOM KERAS LAYER TO LOAD THE MODEL
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package='Custom', name='SingleBiasDense')
class SingleBiasDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, use_bias=True, name=None, **kwargs):
        if name is None:
            name = str(uuid.uuid4())
        super(SingleBiasDense, self).__init__(name=name, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(name='w', shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='b', shape=(1,), initializer='zeros', trainable=True)
        else:
            self.bias = None
        super(SingleBiasDense, self).build(input_shape)

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(SingleBiasDense, self).get_config()
        config.update({'units': self.units, 'activation': tf.keras.activations.serialize(self.activation), 'use_bias': self.use_bias, 'name': self.name})
        return config

# ==============================================================================
# 2. DATA PREPARATION PIPELINE
# ==============================================================================
def extract_hog_features(images, pixels_per_cell=(4, 4), cells_per_block=(3, 3), orientations=10):
    """ Extracts HOG features from a list of images. """
    hog_features = []
    for image in images:
        feature = hog(image.reshape(28, 28), pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, orientations=orientations, block_norm='L2-Hys')
        hog_features.append(feature)
    return np.array(hog_features)

print("\nLoading and processing raw MNIST data...")
# Load and normalize the raw data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

# Extract HOG features
train_hog_features = extract_hog_features(x_train)
test_hog_features = extract_hog_features(x_test)

# Define PCA and fit it ONLY on the training data
N_COMPONENTS_PCA = 16
pca = PCA(n_components=N_COMPONENTS_PCA, svd_solver='arpack')
pca.fit(train_hog_features)

# Transform the test data using the fitted PCA model
x_test_pca = pca.transform(test_hog_features)
print("Data processing complete.")

# ==============================================================================
# 3. LOAD PRE-TRAINED MODEL AND EVALUATE
# ==============================================================================
NUM_CLASSES = 10

try:
    print(f"\nLoading pre-trained model from: {MODEL_EXTRACT_PATH}")
    model = tf.keras.models.load_model(MODEL_EXTRACT_PATH)
    print("Model loaded successfully.")

    # --- 1. EVALUATION (ORIGINAL MODEL) ---
    print("\n--- 1. EVALUATION (ORIGINAL MODEL) ---")
    test_loss, test_accuracy = model.evaluate(
        x_test_pca,
        tf.keras.utils.to_categorical(y_test, NUM_CLASSES),
        verbose=0
    )
    print(f'Test accuracy of the original model: {test_accuracy}')

    # --- 2. EVALUATION (LAYER-WISE PARTITIONED MODEL) ---
    print("\n--- 2. EVALUATION (LAYER-WISE PARTITIONED MODEL) ---")
    print("Splitting model into layer-wise submodels...")
    
    # --- CORRECTED SUBMODEL CREATION LOGIC ---
    # Create submodel 1
    input_layer_1 = tf.keras.layers.Input(shape=(N_COMPONENTS_PCA,))
    output_layer_1 = model.layers[0](input_layer_1)
    submodel_1 = tf.keras.Model(inputs=input_layer_1, outputs=output_layer_1)
    submodel_1.layers[-1].set_weights(model.layers[0].get_weights())

    # Create submodel 2
    input_layer_2 = tf.keras.layers.Input(shape=(submodel_1.output_shape[1],))
    output_layer_2 = model.layers[1](input_layer_2)
    submodel_2 = tf.keras.Model(inputs=input_layer_2, outputs=output_layer_2)
    submodel_2.layers[-1].set_weights(model.layers[1].get_weights())

    # Create submodel 3
    input_layer_3 = tf.keras.layers.Input(shape=(submodel_2.output_shape[1],))
    output_layer_3 = model.layers[2](input_layer_3)
    submodel_3 = tf.keras.Model(inputs=input_layer_3, outputs=output_layer_3)
    submodel_3.layers[-1].set_weights(model.layers[2].get_weights())

    # Create submodel 4
    input_layer_4 = tf.keras.layers.Input(shape=(submodel_3.output_shape[1],))
    output_layer_4 = model.layers[3](input_layer_4)
    submodel_4 = tf.keras.Model(inputs=input_layer_4, outputs=output_layer_4)
    submodel_4.layers[-1].set_weights(model.layers[3].get_weights())
    print("Submodels created.")

    def evaluate_cascade_model(x_test):
        output_1 = submodel_1.predict(x_test, verbose=0)
        output_2 = submodel_2.predict(output_1, verbose=0)
        output_3 = submodel_3.predict(output_2, verbose=0)
        return submodel_4.predict(output_3, verbose=0)

    cascade_predictions = evaluate_cascade_model(x_test_pca)
    cascade_predicted_labels = np.argmax(cascade_predictions, axis=1)
    cascade_accuracy = accuracy_score(y_test, cascade_predicted_labels)
    print(f'Test accuracy of the cascade model: {cascade_accuracy}')

    # --- 3. EVALUATION (INT16 QUANTIZED MODEL) ---
    print("\n--- 3. EVALUATION (INT16 QUANTIZED MODEL) ---")
    print("Applying int16 quantization to model weights...")
    def float_to_signed_int16(weights):
        min_w, max_w = np.min(weights), np.max(weights)
        scale = (max_w - min_w) / (2**15 - 1)
        return np.round((weights - min_w) / scale).astype(np.int16) - 2**15, min_w, scale

    def signed_int16_to_float(q_weights, min_w, scale):
        return (q_weights.astype(np.float32) + 2**15) * scale + min_w

    quantized_model_16bit = tf.keras.models.clone_model(model)
    quantized_model_16bit.build(input_shape=(None, N_COMPONENTS_PCA))

    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            q_kernel, min_w, scale = float_to_signed_int16(layer.kernel.numpy())
            deq_kernel = signed_int16_to_float(q_kernel, min_w, scale.astype(np.float16))
            quantized_model_16bit.layers[i].kernel.assign(deq_kernel)
        if hasattr(layer, 'bias') and layer.bias is not None:
            quantized_model_16bit.layers[i].bias.assign(layer.bias.numpy())

    quantized_model_16bit.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Int16 quantized model compiled.")

    quant16_loss, quant16_accuracy = quantized_model_16bit.evaluate(
        x_test_pca,
        tf.keras.utils.to_categorical(y_test, NUM_CLASSES),
        verbose=0
    )
    print(f'Test accuracy of the int16 quantized model: {quant16_accuracy}')

    # --- 4. EVALUATION (INT8 QUANTIZED MODEL) ---
    print("\n--- 4. EVALUATION (INT8 QUANTIZED MODEL) ---")
    print("Applying int8 quantization to model weights...")
    def float_to_uint8(weights):
        min_w, max_w = np.min(weights), np.max(weights)
        scale = (max_w - min_w) / (2**8 - 1)
        return np.round((weights - min_w) / scale).astype(np.uint8), min_w, scale

    def uint8_to_float(q_weights, min_w, scale):
        return (q_weights.astype(np.float32)) * scale + min_w

    quantized_model_8bit = tf.keras.models.clone_model(model)
    quantized_model_8bit.build(input_shape=(None, N_COMPONENTS_PCA))

    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            q_kernel, min_w, scale = float_to_uint8(layer.kernel.numpy())
            deq_kernel = uint8_to_float(q_kernel, min_w, scale.astype(np.float16))
            quantized_model_8bit.layers[i].kernel.assign(deq_kernel)
        if hasattr(layer, 'bias') and layer.bias is not None:
            # Biases can also be quantized, but are copied here for simplicity
            quantized_model_8bit.layers[i].bias.assign(layer.bias.numpy())

    quantized_model_8bit.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Int8 quantized model compiled.")

    quant8_loss, quant8_accuracy = quantized_model_8bit.evaluate(
        x_test_pca,
        tf.keras.utils.to_categorical(y_test, NUM_CLASSES),
        verbose=0
    )
    print(f'Test accuracy of the int8 quantized model: {quant8_accuracy}')


except Exception as e:
    print(f"An error occurred during model loading or evaluation: {e}")
