import tensorflow as tf
from tensorflow.keras import layers
from parse_tfrec_fns import parse_tfrecord_fn

AUTO = tf.data.experimental.AUTOTUNE 


## Augmentations and Transforms
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
                layers.Resizing(128, 128),
                layers.Rescaling(1./255),
                layers.Normalization(mean = (0.3741, 0.4076, 0.5425), variance = (0.0004, 0.0004, 0.0004))
                ])
    
    image = aug(image)
    
    return image


## Data preprocessing
def prepare_fit_sample(features):
    img_feat = features['image']

    h = tf.shape(img_feat)[0]
    w = tf.shape(img_feat)[1]

    w = tf.cast(w, tf.int64)
    h = tf.cast(h, tf.int64)

    screen_w, screen_h = features['screen_w'], features['screen_h']

    kps = [features['leye_x1']/w, features['leye_y1']/h, features['leye_x2']/w, features['leye_y2']/h,
           features['reye_x1']/w, features['reye_y1']/h, features['reye_x2']/w, features['reye_y2']/h]
    # kps has type float64

    lx, ly, lw, lh = int(features['leye_x']), int(features['leye_y']), int(features['leye_w']), int(features['leye_h'])
    rx, ry, rw, rh = int(features['reye_x']), int(features['reye_y']), int(features['reye_w']), int(features['reye_h'])

    lx = tf.clip_by_value(lx, 0, int(w)-lw)
    ly = tf.clip_by_value(ly, 0, int(h)-lh)

    rx = tf.clip_by_value(rx, 0, int(w)-rw)
    ry = tf.clip_by_value(ry, 0, int(h)-rh)

    l_eye = tf.image.crop_to_bounding_box(img_feat, ly, lx, lh, lw)
    r_eye = tf.image.crop_to_bounding_box(img_feat, ry, rx, rh, rw)

    l_eye = tf.image.flip_left_right(l_eye)

    l_eye = augmentation(l_eye)
    r_eye = augmentation(r_eye)

    y = [features['dot_xcam'], features['dot_y_cam']]
    # y has type float32

    return (l_eye, r_eye, kps), y


def prepare_eval_sample(features):
    img_feat = features['image']

    h = tf.shape(img_feat)[0]
    w = tf.shape(img_feat)[1]

    w = tf.cast(w, tf.int64)
    h = tf.cast(h, tf.int64)

    screen_w, screen_h = features['screen_w'], features['screen_h']

    kps = [features['leye_x1']/w, features['leye_y1']/h, features['leye_x2']/w, features['leye_y2']/h,
           features['reye_x1']/w, features['reye_y1']/h, features['reye_x2']/w, features['reye_y2']/h]
    # kps has type float64

    lx, ly, lw, lh = int(features['leye_x']), int(features['leye_y']), int(features['leye_w']), int(features['leye_h'])
    rx, ry, rw, rh = int(features['reye_x']), int(features['reye_y']), int(features['reye_w']), int(features['reye_h'])

    lx = tf.clip_by_value(lx, 0, int(w)-lw)
    ly = tf.clip_by_value(ly, 0, int(h)-lh)

    rx = tf.clip_by_value(rx, 0, int(w)-rw)
    ry = tf.clip_by_value(ry, 0, int(h)-rh)

    l_eye = tf.image.crop_to_bounding_box(img_feat, ly, lx, lh, lw)
    r_eye = tf.image.crop_to_bounding_box(img_feat, ry, rx, rh, rw)

    l_eye = tf.image.flip_left_right(l_eye)

    l_eye = augmentation(l_eye, False)
    r_eye = augmentation(r_eye, False)

    y = [features['dot_xcam'], features['dot_y_cam']]
    # y has type float32

    return (l_eye, r_eye, kps), y



## Creating TF dataloader
def get_fit_dataset(filenames, batch_size):
    option_no_order = tf.data.Options()
    option_no_order.deterministic = False  # disable order, increase speed
    
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        .with_options(option_no_order)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTO)
        .map(prepare_fit_sample, num_parallel_calls=AUTO)
        .shuffle(batch_size*10)
        .batch(batch_size)
        .prefetch(buffer_size=AUTO)
    )
    
    return dataset


def get_eval_dataset(filenames, batch_size):
    option_no_order = tf.data.Options()
    option_no_order.deterministic = False  # disable order, increase speed
    
    dataset = (
        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        .with_options(option_no_order)
        .map(parse_tfrecord_fn, num_parallel_calls=AUTO)
        .map(prepare_eval_sample, num_parallel_calls=AUTO)
        .shuffle(batch_size*10)
        .batch(batch_size)
        .prefetch(buffer_size=AUTO)
    )
    
    return dataset