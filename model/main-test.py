from comet_ml import Experiment
import tensorflow as tf
from get_batched_dataset import get_fit_dataset, get_eval_dataset
from model import gazetrack_model
from model_params import lr, loss, metrics, optimizer, scheduler

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
        auto_metric_logging=False,
        auto_param_logging=False,
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
            'scheduler':scheduler
            }

    
    train_dataset = get_fit_dataset(args.dataset_dir+'train.tfrec', args.batch_size)
    valid_dataset = get_fit_dataset(args.dataset_dir+'val.tfrec', args.batch_size)
    test_dataset = get_eval_dataset(args.dataset_dir+'test.tfrec', args.batch_size)

    print('->datasets created')
    
    save_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=args.save_dir+'model--{epoch:02d}-{val_loss:.3f}.h5') 

    
    model = gazetrack_model()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,)

    print('->model compiled')
    
    with experiment.train():
        history = model.fit(x=train_dataset,   
                            batch_size=args.batch_size,
                            epochs=args.epochs,  
                            verbose=2,   #auto=1, 1=progress bar, 2=one line per epoch( maybe use 2 if running job)
                            callbacks=[scheduler,save_ckpt],
                            validation_data=valid_dataset,
                            shuffle=True,    #probably will not work as our dataset is a tf.data object
                            initial_epoch=0,     #epoch at which to resume training
                            workers=1,
                            use_multiprocessing=False
                            )

    
    print('->training done')

    with experiment.test():
        loss, accuracy = model.evaluate(x=test_dataset,
                                        batch_size=args.batch_size,
                                        verbose=2
                                        )
        metrics = {
            'loss':loss,
            'accuracy':accuracy
        }
        experiment.log_metrics(metrics)

    print('->eval done')

    experiment.log_parameters(params) 

    print('->logging params done')