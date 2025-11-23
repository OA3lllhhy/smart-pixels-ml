
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model
from OptimizedDataGenerator_v2 import OptimizedDataGenerator
from models import CreateModel

epochs = 5000
batch_size = 5000
learning_rate = 0.001
early_stopping_patience = 50
shape = (13, 21, 8)


stamp = "08ts5000"
base_path = f"/ceph/submit/data/user/h/haoyun22/smart_pixels_data/tfrecords_{stamp}"

tfrecords_train = f"{base_path}/train"
tfrecords_val   = f"{base_path}/val"
tfrecords_test  = f"{base_path}/test"

checkpoint_directory = Path(f"./checkpoints_{stamp}_Q")

test_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir = tfrecords_test,
    shuffle = False,
    seed = 13,
    quantize = True
)

# --------------------------------------------------------
#  SELECT FINAL WEIGHTS
# --------------------------------------------------------
weights_path = sorted(checkpoint_directory.glob("weights*.hdf5"))[-1]
print(f"Using final weights: {weights_path}")


# --------------------------------------------------------
#  EVALUATION
# --------------------------------------------------------
outfile_directory = Path(f"./outfile_{stamp}_Q")
outfile_directory.mkdir(parents=True, exist_ok=True)

outfile_name = f"evaluation_{weights_path.stem}_Q.csv"
outfile_path = outfile_directory / outfile_name

# Load best model
model = CreateModel(shape=shape, n_filters=5, pool_size=3)
model.load_weights(weights_path)

# Predict
p_test = model.predict(test_generator)

# Collect true labels
complete_truth = None
for _, y in test_generator:
    if complete_truth is None:
        complete_truth = y
    else:
        complete_truth = np.concatenate((complete_truth, y), axis=0)

# Build results table
df = pd.DataFrame(
    p_test,
    columns=['x','M11','y','M22','cotA','M33','cotB','M44',
             'M21','M31','M32','M41','M42','M43']
)

df["xtrue"], df["ytrue"], df["cotAtrue"], df["cotBtrue"] = complete_truth.T

# Clamp matrix diagonals ≥ 0
for m in ["M11","M22","M33","M44"]:
    df[m] = np.maximum(df[m], 1e-9)

# Residuals
df["residual_x"] = df["xtrue"] - df["x"]
df["residual_y"] = df["ytrue"] - df["y"]
df["residual_A"] = df["cotAtrue"] - df["cotA"]
df["residual_B"] = df["cotBtrue"] - df["cotB"]

df.to_csv(outfile_path, index=False)

print(f"✓ Evaluation saved to {outfile_path}")