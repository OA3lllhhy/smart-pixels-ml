from OptimizedDataGenerator_v2 import OptimizedDataGenerator
import glob, os

# -----------------------------------
# Paths
# -----------------------------------

time_slice = int(8)
batch_size = int(5000)

dataset_base = "/ceph/submit/data/user/a/anton100/output_v3"
stamp = f"0{time_slice}ts{batch_size}"
print(f"Generating TFRecords with stamp: {stamp}")

out_base = f"/ceph/submit/data/user/h/haoyun22/smart_pixels_data/tfrecords_{stamp}"
os.makedirs(out_base, exist_ok=True)

# -----------------------------------
# Detect number of shards
# -----------------------------------
parts = sorted(glob.glob(f"{dataset_base}/part.*.parquet"))
N = 70 #len(parts)

n_train = int(0.6 * N)
n_val   = int(0.2 * N)
n_test  = N - n_train - n_val

print("Total:", N)
print("Train:", n_train)
print("Val:", n_val)
print("Test:", n_test)

# -----------------------------------
# TFRecord directories
# -----------------------------------
train_dir = f"{out_base}/train"
val_dir   = f"{out_base}/val"
test_dir  = f"{out_base}/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# -----------------------------------
# Common parameters
# -----------------------------------
batch_size = 5000
input_shape = (time_slice, 13, 21)
transpose   = (0, 2, 3, 1)

# -----------------------------------
# TRAIN = first 60%
# -----------------------------------
gen_train = OptimizedDataGenerator(
    dataset_base_dir = dataset_base,
    batch_size = batch_size,
    file_count = n_train,
    files_from_end = False,
    input_shape = input_shape,
    transpose = transpose,
    use_time_stamps = list(range(time_slice)),
    to_standardize = True,
    shuffle = False,
    max_workers = 4,
    tfrecords_dir = train_dir,
)
print("✓ TRAIN TFRecords created.")

# -----------------------------------
# VALIDATION = NEXT 20%
# -----------------------------------
gen_val = OptimizedDataGenerator(
    dataset_base_dir = dataset_base,
    batch_size = batch_size,
    file_count = n_val,
    files_from_end = False,
    input_shape = input_shape,
    transpose = transpose,
    use_time_stamps = list(range(time_slice)),
    to_standardize = True,
    shuffle = False,
    max_workers = 4,
    tfrecords_dir = val_dir,
)

# Manually shift the validation to start AFTER training
gen_val.files = gen_val.files[n_train : n_train + n_val]
print("✓ VALIDATION TFRecords created.")

# -----------------------------------
# TEST = last 20%
# -----------------------------------
gen_test = OptimizedDataGenerator(
    dataset_base_dir = dataset_base,
    batch_size = batch_size,
    file_count = n_test,
    files_from_end = True,
    input_shape = input_shape,
    transpose = transpose,
    use_time_stamps = list(range(time_slice)),
    to_standardize = True,
    shuffle = False,
    max_workers = 4,
    tfrecords_dir = test_dir,
)
print("✓ TEST TFRecords created.")