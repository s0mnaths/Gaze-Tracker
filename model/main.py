import argparse
import comet_ml
experiment = comet_ml.Experiment("YOUR-API-KEY")



parser = argparse.ArgumentParser(description='Train GazeTracker TensorFlow/Keras')
parser.add_argument('--dataset_dir', default='../datasets/gazetrack_tfrec', help='Path to TFRecord dataset')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--save_dir', default='../checkpoints/', help='Path to store checkpoints')
parser.add_argument('--comet_name', default='../gazetrack-cml', help='Path to store checkpoints')
parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs to use')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--checkpoint', default=None, help='Path to load pre-trained weights')



if __name__ == '__main__':
    args = parser.parse_args()
    proj_name = args.comet_name
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, filename='{epoch}-{val_loss:.3f}-{train_loss:.3f}', save_top_k=-1)
    logger = CometLogger(
        api_key="YOUR-API-KEY",
        project_name=proj_name,
    )
    
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():

        model = gazetrack_model()
        print(model.summary())

        # COMPILE
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    # FIT
    model.fit(
        x=get_dataset(train_filenames, batch_size),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
    )

    # SAVE
    model.save(args.model_output + '/1')