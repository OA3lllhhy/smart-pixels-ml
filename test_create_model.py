import qkeras
import tensorflow as tf
print(f"QKeras version: {qkeras.__version__}")
print(f"TensorFlow version: {tf.__version__}")

from qkeras import QDense, quantized_bits
from keras.layers import Input
from keras.models import Model

# Simple test
x_in = Input(shape=(10,))
x = QDense(5, kernel_quantizer=quantized_bits(8, 0, alpha=1))(x_in)
model = Model(inputs=x_in, outputs=x)
model.build((None, 10))
print(f"Test model parameters: {model.count_params()}")
model.summary()