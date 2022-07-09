import tensorflow as tf
from tensorflow.keras import layers
from create_tfrec import parse_tfrecord_fn

AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

TRAINING_FILENAMES = '../datasets/gazetrack_tfrec/train.tfrec'
VALID_FILENAMES = '../datasets/gazetrack_tfrec/val.tfrec'
TEST_FILENAMES = '../datasets/gazetrack_tfrec/test.tfrec'
BATCH_SIZE = 256

def augmentation(image, training = True):
    if training:
        aug = tf.keras.Sequential([
                layers.Resizing(128+10, 128+10),
                layers.RandomCrop(128, 128, 256),
                layers.Rescaling(1./255),
                layers.Normalization(mean = (0.3741, 0.4076, 0.5425), variance = (0.0004, 0.0004, 0.0004))
                ])
        
    else:
        aug = tf.keras.Sequential([
                layers.Resizing(128+10, 128+10),
                layers.Rescaling(1./255),
                layers.Normalization(mean = (0.3741, 0.4076, 0.5425), variance = (0.0004, 0.0004, 0.0004))
                ])
    
    image = aug(image)
    
    return image


def prepare_sample(features):
    image = features['image']
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    
    w = tf.cast(w, tf.int64)
    h = tf.cast(h, tf.int64)
    
    screen_w, screen_h = features['screen_w'], features['screen_h']
    
    kps = [features['leye_x1']/w, features['leye_y1']/h, features['leye_x2']/w, features['leye_y2']/h,
           features['reye_x1']/w, features['reye_y1']/h, features['reye_x2']/w, features['reye_y2']/h]
    # kps has type float64
    

    lx, ly, lw, lh = features['leye_x'], features['leye_y'], features['leye_w'], features['leye_h']
    rx, ry, rw, rh = features['reye_x'], features['reye_y'], features['reye_w'], features['reye_h']
    
    lx = tf.cast(lx, tf.int32)
    ly = tf.cast(ly, tf.int32)
    lw = tf.cast(lw, tf.int32)
    lh = tf.cast(lh, tf.int32)
    
    rx = tf.cast(rx, tf.int32)
    ry = tf.cast(ry, tf.int32)
    rw = tf.cast(rw, tf.int32)
    rh = tf.cast(rh, tf.int32)
    
    l_eye = tf.image.crop_to_bounding_box(image, ly, lx, lh, lw)  
    r_eye = tf.image.crop_to_bounding_box(image, ry, rx, rh, rw)
    
    l_eye = tf.image.flip_left_right(l_eye)
    
    out = [features['dot_xcam'], features['dot_y_cam']]
    # out has type float32
    
    l_eye = augmentation(l_eye)
    r_eye = augmentation(r_eye)
    
    
    return l_eye, r_eye, kps, out, screen_w, screen_h



def get_batched_dataset(filenames, batch_size):
    option_no_order = tf.data.Options()
    option_no_order.deterministic = False  # disable order, increase speed
    
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        .with_options(option_no_order)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTO)
        .map(prepare_sample, num_parallel_calls=AUTO)
        .shuffle(batch_size*10)
        .batch(batch_size)
        .prefetch(buffer_size=AUTO)
    )
    
    return dataset


train_dataset = get_batched_dataset(TRAINING_FILENAMES, BATCH_SIZE)
valid_dataset = get_batched_dataset(VALID_FILENAMES, BATCH_SIZE)
test_dataset = get_batched_dataset(TEST_FILENAMES, BATCH_SIZE)

train_len = sum(1 for _ in tf.data.TFRecordDataset(TRAINING_FILENAMES))
val_len = sum(1 for _ in tf.data.TFRecordDataset(VALID_FILENAMES))
test_len = sum(1 for _ in tf.data.TFRecordDataset(TEST_FILENAMES))

print(f"No. of train samples: {train_len}")
print(f"No. of val samples: {val_len}")
print(f"No. of test samples: {test_len}")
