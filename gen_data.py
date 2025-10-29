import os
import random
from pathlib import Path
import time
import argparse
import json
import submitit
import shutil
from OptimizedDataGenerator import OptimizedDataGenerator
from loss import custom_loss
from models import CreateModel

epochs = 200
batch_size = 128
val_batch_size = 128
train_file_size = 5
val_file_size = 2
n_filters = 5
pool_size = 3
learning_rate = 0.001
early_stopping_patience = 15

# paths
data_directory_path = "/home/hep/hl2822/smart-pixels-ml/datasets/recon3D/"
labels_directory_path = "/home/hep/hl2822/smart-pixels-ml/datasets/labels/"

# Create tf record datasets
# stamp = '%08x' % random.randrange(16**8)
stamp = "d7414f9d"
print(f"Using stamp: {stamp}")
output_directory = Path(f"./tfrecords_{stamp}").resolve()
os.makedirs(output_directory, exist_ok=True)
tfrecords_dir_train = Path(output_directory, f"tfrecords_train_{stamp}").resolve()
tfrecords_dir_validation = Path(output_directory, f"tfrecords_validation_{stamp}").resolve()

print("Creating 3D training data generator...")
start_time = time.time()

validation_generator = OptimizedDataGenerator(
    data_directory_path=data_directory_path,
    labels_directory_path=labels_directory_path,
    is_directory_recursive=False,
    file_type="parquet",
    data_format="3D",
    batch_size=val_batch_size,
    file_count=val_file_size,
    to_standardize=True,
    include_y_local=False,
    labels_list=['x-midplane', 'y-midplane', 'cotAlpha', 'cotBeta'],
    input_shape=(20, 13, 21),  # All 20 time slices
    transpose=(0, 2, 3, 1),
    files_from_end=True,
    save=True,
    use_time_stamps=list(range(20)),  # Use all time stamps
    tfrecords_dir=tfrecords_dir_validation,
)

print(f"✓ 3D Training generator created in {time.time() - start_time:.2f} seconds")
print(f"Input shape per sample: (20, 13, 21)")