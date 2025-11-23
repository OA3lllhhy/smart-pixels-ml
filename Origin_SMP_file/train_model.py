# fix for keras v3.0 update
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' 

# tensorflow
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import Sequence
from qkeras import *


# python based
import random
from pathlib import Path
import time
import argparse
import json
import submitit
import shutil

# custom code
# from dataloaders.OptimizedDataGenerator import OptimizedDataGenerator
from OptimizedDataGenerator import OptimizedDataGenerator
from loss import custom_loss
# from models import CreateModel

# set gpu growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class ModelBuilder:
    
    # @staticmethod
    # def var_network(var, hidden=10, output=2):
    #     """
    #     Fully connected network with quantized layers
        
    #     Args:
    #         var: input tensor
    #         hidden: number of hidden units
    #         output: number of output units
            
    #     Returns:
    #         output tensor
    #     """
    #     var = Flatten()(var)
    #     var = QDense(
    #         hidden,
    #         kernel_quantizer=quantized_bits(8, 0, alpha=1),
    #         bias_quantizer=quantized_bits(8, 0, alpha=1),
    #         kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    #         activity_regularizer=tf.keras.regularizers.L2(0.01),
    #     )(var)
    #     var = QActivation("quantized_tanh(8, 0, 1)")(var)
    #     var = QDense(
    #         hidden,
    #         kernel_quantizer=quantized_bits(8, 0, alpha=1),
    #         bias_quantizer=quantized_bits(8, 0, alpha=1),
    #         kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    #         activity_regularizer=tf.keras.regularizers.L2(0.01),
    #     )(var)
    #     var = QActivation("quantized_tanh(8, 0, 1)")(var)
    #     return QDense(
    #         output,
    #         kernel_quantizer=quantized_bits(8, 0, alpha=1),
    #         bias_quantizer=quantized_bits(8, 0, alpha=1),
    #         kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    #     )(var)
    
    @staticmethod
    def var_network(var, hidden=10, output=2):
        """
        Fully connected network with standard Keras layers
        
        Args:
            var: input tensor
            hidden: number of hidden units
            output: number of output units
            
        Returns:
            output tensor
        """
        var = Flatten()(var)
        var = Dense(
            hidden,
            activation='tanh',
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
        )(var)
        var = Dense(
            hidden,
            activation='tanh',
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
        )(var)
        return Dense(
            output,
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        )(var)
    
    # @staticmethod
    # def conv_network(var, n_filters=5, kernel_size=3):
    #     """
    #     2D convolutional network with quantized layers
        
    #     Args:
    #         var: input tensor
    #         n_filters: number of convolutional filters
    #         kernel_size: size of convolutional kernel
            
    #     Returns:
    #         output tensor
    #     """
    #     var = QSeparableConv2D(
    #         n_filters, kernel_size,
    #         depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
    #         pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
    #         bias_quantizer=quantized_bits(4, 0, alpha=1),
    #         depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
    #         pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
    #         activity_regularizer=tf.keras.regularizers.L2(0.01),
    #     )(var)
    #     var = QActivation("quantized_tanh(4, 0, 1)")(var)
    #     var = QConv2D(
    #         n_filters, 1,
    #         kernel_quantizer=quantized_bits(4, 0, alpha=1),
    #         bias_quantizer=quantized_bits(4, 0, alpha=1),
    #         kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    #         activity_regularizer=tf.keras.regularizers.L2(0.01),
    #     )(var)
    #     var = QActivation("quantized_tanh(4, 0, 1)")(var)    
    #     return var

    @staticmethod
    def conv_network(var, n_filters=5, kernel_size=3):
        """
        2D convolutional network with standard Keras layers
        
        Args:
            var: input tensor
            n_filters: number of convolutional filters
            kernel_size: size of convolutional kernel
            
        Returns:
            output tensor
        """
        var = SeparableConv2D(
            n_filters, 
            kernel_size,
            activation='tanh',
            padding='same',
            depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
            pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
        )(var)
        var = Conv2D(
            n_filters, 
            1,
            activation='tanh',
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
        )(var)
        return var

    @staticmethod
    def conv_network_3d(var, n_filters=5, kernel_size=3):
        """
        3D convolutional network across time, x, y dimensions
        
        Args:
            var: input tensor
            n_filters: number of convolutional filters
            kernel_size: size of convolutional kernel
            
        Returns:
            output tensor
        """
        # 3D convolution across time, x, y dimensions using Keras Conv3D
        var = Conv3D(
            n_filters, 
            kernel_size=(kernel_size, kernel_size, kernel_size),  # (time, x, y)
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
            padding='same'
        )(var)
        var = QActivation("quantized_tanh(4, 0, 1)")(var)
        var = Conv3D(
            n_filters, 
            kernel_size=(1, 1, 1),  # 1x1x1 conv for channel mixing
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
        )(var)
        var = QActivation("quantized_tanh(4, 0, 1)")(var)    
        return var

    @classmethod
    def CreateModel(cls, shape, n_filters=5, pool_size=3):
        x_base = x_in = Input(shape)
        stack = cls.conv_network(x_base, n_filters=n_filters)
        stack = AveragePooling2D(pool_size=(pool_size, pool_size),
                                 strides=None,
                                 padding='valid',
                                 data_format=None
                                )(stack)
        # stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
        stack = Activation("linear")(stack)
        stack = cls.var_network(stack, hidden=16, output=14)
        model = Model(inputs=x_in, outputs=stack)
        return model



def train(
    output_directory = Path(f"/home/submit/haoyun22/smart-pixels-ml/train_model_output").resolve(),
    epochs = 200,
    batch_size = 128,
    val_batch_size = 128,
    train_file_size = 10, # controls number of train files used -> seem to run into a problem using >=50 files maybe with memory
    val_file_size = 4, # controls number of validation files used
    n_filters = 5, # model number of filters
    pool_size = 3, # model pool size
    learning_rate = 0.001, 
    early_stopping_patience = 15,
):
    
    # update %j with actual job number
    try:
        job_env = submitit.JobEnvironment()
        output_directory = Path(str(output_directory).replace("%j", str(job_env.job_id)))
    except:
        output_directory = Path(str(output_directory).replace("%j", "%08x" % random.randrange(16**8)))
    os.makedirs(output_directory, exist_ok=True)
    print(output_directory)

    # paths
    data_directory_path = "/ceph/submit/data/user/a/anton100/datasets/recon3D/" # "/net/scratch/badea/dataset8/unflipped/"
    labels_directory_path = "/ceph/submit/data/user/a/anton100/datasets/labels/" # "/net/scratch/badea/dataset8/unflipped/"
    
    # create tf records directory
    # stamp = 'd7414f9d' % random.randrange(16**8)
    # tfrecords_dir_train = Path(output_directory, f"tfrecords_train_{stamp}").resolve()
    # tfrecords_dir_validation = Path(output_directory, f"tfrecords_validation_{stamp}").resolve()

    # training generator
    start_time = time.time()
    # training_generator = OptimizedDataGenerator(
    #     data_directory_path = data_directory_path,
    #     labels_directory_path = labels_directory_path,
    #     is_directory_recursive = False,
    #     file_type = "parquet",
    #     data_format = "3D",
    #     batch_size = batch_size,
    #     file_count = train_file_size,
    #     to_standardize= True,
    #     include_y_local= False,
    #     labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
    #     input_shape = (2,13,21), # (20,13,21),
    #     transpose = (0,2,3,1),
    #     save=True,
    #     use_time_stamps = [0,19],
    #     tfrecords_dir = tfrecords_dir_train,
    # )
    training_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir = "/ceph/submit/data/user/h/haoyun22/smart_pixels_data/tfrecords_d7414f9d/tfrecords_train_d7414f9d/",
    shuffle = True,
    seed = 13,
    quantize = False
    )

    print("--- Training generator %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    # validation_generator = OptimizedDataGenerator(
    #     data_directory_path = data_directory_path,
    #     labels_directory_path = labels_directory_path,
    #     is_directory_recursive = False,
    #     file_type = "parquet",
    #     data_format = "3D",
    #     batch_size = val_batch_size,
    #     file_count = val_file_size,
    #     to_standardize= True,
    #     include_y_local= False,
    #     labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
    #     input_shape = (2,13,21), # (20,13,21),
    #     transpose = (0,2,3,1),
    #     files_from_end=True,
    #     use_time_stamps = [0,19],
    #     tfrecords_dir = tfrecords_dir_validation,
    # )
    validation_generator = OptimizedDataGenerator(
    load_from_tfrecords_dir = "/ceph/submit/data/user/h/haoyun22/smart_pixels_data/tfrecords_d7414f9d/tfrecords_validation_d7414f9d/",
    shuffle = True,
    seed = 13,
    quantize = False
    )

    print("--- Validation generator %s seconds ---" % (time.time() - start_time))

    # print("\n=== Testing simple Keras model ===")
    # from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
    # from tensorflow.keras.models import Model
    
    # test_input = Input(shape=(13,21,2))
    # test_x = Conv2D(5, 3, activation='relu')(test_input)
    # test_x = Flatten()(test_x)
    # test_x = Dense(4)(test_x)
    # model = Model(inputs=test_input, outputs=test_x)
    # print(f"Simple model parameters: {model.count_params():,}")
    # model.summary()
    # model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)

    # compiles model
    start_time = time.time()
    model = ModelBuilder.CreateModel(shape=(13,21,20), n_filters=5, pool_size=3)
    print(f"\nModel has {model.count_params():,} parameters")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)
    model.summary()
    print("--- Model create and compile %s seconds ---" % (time.time() - start_time))

    # launch quick training once gpu is available
    es = EarlyStopping(
        patience=early_stopping_patience,
        restore_best_weights=True
    )
    
    # checkpoint path
    checkpoint_filepath = Path(output_directory, 'weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5').resolve()
    mcp = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=False,
    )

    # logger
    csv_logger = CSVLogger(Path(output_directory, 'training_log.csv').resolve())

    # train
    history = model.fit(x=training_generator,
                        validation_data=validation_generator,
                        callbacks=[mcp, csv_logger, es],
                        epochs=epochs,
                        shuffle=False, # shuffling now occurs within the data-loader
                        verbose=1)
    
    # clean up tf records
    # shutil.rmtree(tfrecords_dir_train)
    # shutil.rmtree(tfrecords_dir_validation)

if __name__ == "__main__":

    # set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="path to json file containing query", default=None)
    parser.add_argument("--njobs", help="number of jobs to actually launch. default is all", default=-1, type=int)
    parser.add_argument("-e", "--epochs", help="number of epochs to train for", default=1, type=int)
    args = parser.parse_args()
    
    # read in query
    # if Path(args.query).resolve().exists():
    #     query_path = Path(args.query).resolve()
    # else:
    #     # throw
    #     raise ValueError(f"Could not locate {args.query} in query directory or as absolute path")
    # with open(query_path) as f:
    #     query = json.load(f)

    # read in query
    if args.query is None:
        query = {"submitit": False}  # Default: run locally without submitit
    elif Path(args.query).resolve().exists():
        query_path = Path(args.query).resolve()
        with open(query_path) as f:
            query = json.load(f)
    else:
        raise ValueError(f"Could not locate {args.query}")

    # create top level output directory
    top_dir = Path("results", f'./training-{"%08x" % random.randrange(16**8)}', "%j").resolve()

    # create some configurations
    confs = []
    for n_filters in [1,2,3,4,5]:
       for pool_size in [1,2,3,4,5]:
            confs.append({
                "n_filters" : n_filters,
                "pool_size" : pool_size,
                "output_directory" : Path(top_dir, f'./weights-nFilters{n_filters}-poolSize{pool_size}-checkpoints').resolve(),
                "epochs" : args.epochs
            })

    # if submitit false then just launch job
    if not query.get("submitit", False):
        for iC, conf in enumerate(confs):
            # only launch a single job
            if args.njobs != -1 and (iC+1) > args.njobs:
                continue
            print(conf)
            train(**conf)
        exit()
    

    # submission
    executor = submitit.AutoExecutor(folder=top_dir)
    executor.update_parameters(**query.get("slurm", {}))
    # the following line tells the scheduler to only run at most 2 jobs at once. By default, this is several hundreds
    # executor.update_parameters(slurm_array_parallelism=2)
    
    # loop over configurations
    jobs = []
    with executor.batch():
        for iC, conf in enumerate(confs):
            
            # only launch a single job
            if args.njobs != -1 and (iC+1) > args.njobs:
                continue
            
            print(conf)

            # if submitit is true in our query json, we'll use submitit
            # if query.get("submitit", False):
            job = executor.submit(train, **conf)
            jobs.append(job)
            # else:
            #     train(**conf)
