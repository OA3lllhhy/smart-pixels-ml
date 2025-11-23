import os
import pathlib
import random
import json
import submitit
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from OptimizedDataGenerator_v2 import OptimizedDataGenerator
from models import CreateModel
from loss import custom_loss

# --------------------------------------------------------
#  HYPERPARAMETERS
# --------------------------------------------------------

epochs = 500
batch_size = 5000
learning_rate = 0.001
early_stopping_patience = 50

# NEW SHAPE: 20 timestamps, 13x21 pixel plane → after transpose becomes (13,21,20)
shape = (13, 21, 8)

# stamp = "2ts5000"
stamp = "08ts5000"
base_path = f"/ceph/submit/data/user/h/haoyun22/smart_pixels_data/tfrecords_{stamp}"

tfrecords_train = f"{base_path}/train"
tfrecords_val   = f"{base_path}/val"
tfrecords_test  = f"{base_path}/test"

# --------------------------------------------------------
#  LOAD DATA WITH OPTIMIZED DATA GENERATOR V2
# --------------------------------------------------------
training_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir = tfrecords_train,
    shuffle = True,
    seed = 13,
    quantize = True
)

validation_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir = tfrecords_val,
    shuffle = True,
    seed = 13,
    quantize = True
)

# --------------------------------------------------------
#  BUILD MODEL
# --------------------------------------------------------
model = CreateModel(shape=shape, n_filters=5, pool_size=3)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)
model.summary()


# --------------------------------------------------------
#  CHECKPOINTS + LOGGING
# --------------------------------------------------------
checkpoint_directory = Path(f"./checkpoints_{stamp}_Q")
checkpoint_directory.mkdir(parents=True, exist_ok=True)

checkpoint_filepath = checkpoint_directory / "weights.{epoch:03d}-t{loss:.3f}-v{val_loss:.3f}.hdf5"
mcp = ModelCheckpoint(
    filepath = str(checkpoint_filepath),
    save_weights_only = True,
    monitor = "val_loss",
    save_best_only = False,
)

csvlogger_directory = Path("./csvlogs")
csvlogger_directory.mkdir(parents=True, exist_ok=True)

csv_logger = CSVLogger(str(csvlogger_directory / f"training_log_{stamp}_Q.csv"), append=True)

es = EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)

# --------------------------------------------------------
#  TRAIN
# --------------------------------------------------------

checkpoint_directory = Path(f"./checkpoints_{stamp}_Q")
checkpoint_files = sorted(checkpoint_directory.glob("weights.*.hdf5"))
if checkpoint_files:
    latest_checkpoint = str(checkpoint_files[-1])
    print(f"Loading weights from {latest_checkpoint}")
    model.load_weights(latest_checkpoint)
    
    import re
    match = re.search(r'weights\.(\d+)-t', latest_checkpoint)
    if match:
        initial_epoch = int(match.group(1))
        print(f"Resuming from epoch {initial_epoch}")
    else:
        initial_epoch = 0
else:
    print("No checkpoint found, starting fresh training.")
    initial_epoch = 0

history = model.fit(
    x = training_generator,
    validation_data = validation_generator,
    epochs = epochs,
    shuffle = False,       # internal shuffling already done by generator
    callbacks = [es, mcp, csv_logger],
    verbose = 1,
    initial_epoch = initial_epoch
)

print("✓ Training complete.")