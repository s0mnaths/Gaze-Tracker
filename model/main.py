from comet_ml import Experiment
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from get_batched_dataset import get_fit_dataset, get_eval_dataset
from new_model import gazetrack_model
from model_params import lr, loss, metrics, optimizer

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


import argparse

print('->import done')

parser = argparse.ArgumentParser(description='Train GazeTracker TensorFlow/Keras')
parser.add_argument('--dataset_dir', default='../datasets/gazetrack_tfrec/', help='Path to TFRecord dataset')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--save_dir', default='../checkpoints/', help='Path to store checkpoints')
parser.add_argument('--comet_name', default='../gazetrack-cml', help='Path to store checkpoints')
parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs to use')  ####
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--checkpoint', default=None, help='Path to load pre-trained weights')
parser.add_argument('--version_description', default=None, help='Description of version')


if __name__ == '__main__':
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # physical_devices = tf.config.list_physical_devices('GPU')
    # for device in physical_devices:
    #     tf.config.experimental.set_memory_growth(device, True)

    print('->no. of gpus printed')
    
    args = parser.parse_args()

    experiment = Experiment(
        api_key="jf5htRXdnj6QxcdXJyvvzPYJg",
        project_name="test1", 
        workspace="s0mnaths",
        auto_metric_logging=True,
        auto_param_logging=True,
        log_graph=True,
        auto_metric_step_rate=True,
        parse_args=True,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_histogram_activation_logging=True,
        auto_histogram_epoch_rate=True,
    )

    print('->experiment initted')


    params={'batch_size':args.batch_size,
            'epochs':args.epochs,
            'optimizer':optimizer,
            'loss':loss,
            'description':args.version_description
            }

    
    train_dataset = get_fit_dataset(args.dataset_dir+'train.tfrec', args.batch_size)
    valid_dataset = get_eval_dataset(args.dataset_dir+'val.tfrec', args.batch_size)
    test_dataset = get_eval_dataset(args.dataset_dir+'test.tfrec', args.batch_size)

    print('->datasets created')
    
    
    save_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=args.save_dir+'epoch-{epoch:02d}-vl-{val_loss:.3f}.ckpt', save_format='tf') 

    model_ = gazetrack_model()
    input1 = Input(shape=(128,128,3))
    input2 = Input(shape=(128,128,3))
    input3 = Input(shape=(8, ))
    combine_input = [input1, input2, input3]
    gz_output = model_(combine_input)

    dense2 = layers.Dense(4)(gz_output)
    bn2 = layers.BatchNormalization(axis = -1, momentum=0.9)(dense2)
    relu = layers.ReLU()(bn2)
    dense3 = layers.Dense(2)(relu)

    final_model = Model(inputs=combine_input, outputs=dense3)
    
    # model = gazetrack_model()
    final_model.compile(optimizer=optimizer, loss=loss, metrics=metrics,)

    print('->model compiled')
    
    with experiment.train():
        history = final_model.fit(x=train_dataset,   
                            batch_size=args.batch_size,
                            epochs=args.epochs,  
                            verbose=2,   #auto=1, 1=progress bar, 2=one line per epoch( maybe use 2 if running job)
                            callbacks=[save_ckpt],
                            validation_data=valid_dataset,
                            shuffle=True,    #probably will not work as our dataset is a tf.data object
                            initial_epoch=0,     #epoch at which to resume training
                            workers=1,
                            use_multiprocessing=False
                            )

    
    print('->training done')

    with experiment.test():
        loss, rmse = final_model.evaluate(x=test_dataset,
                                        batch_size=args.batch_size,
                                        verbose=2
                                        )
        metrics = {
            'loss':loss,
            'rmse':rmse
        }
        experiment.log_metrics(metrics)

    print('->eval done')

    experiment.log_parameters(params) 

    print('->logging params done')