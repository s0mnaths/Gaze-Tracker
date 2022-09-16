# from comet_ml import Experiment
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
import numpy as np 
import matplotlib.pyplot as plt
import json
import os 
from PIL import Image

print('-> import done')

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

print('-> experiment initted')

# AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

# TRAINING_FILENAMES = './datasets/gazetrack_tfrec/train.tfrec' 
# VALID_FILENAMES = './datasets/gazetrack_tfrec/val.tfrec'
# TEST_FILENAMES = './datasets/gazetrack_tfrec/test.tfrec' 
# BATCH_SIZE = 256

# SEED = tf.Variable(256)


# def parse_tfrecord_fn(example):
#     feature_description = {
#         "image": tf.io.FixedLenFeature([], tf.string),
#         "path": tf.io.FixedLenFeature([], tf.string),
#         "device": tf.io.FixedLenFeature([], tf.string),
#         "screen_h": tf.io.FixedLenFeature([], tf.int64),
#         "screen_w": tf.io.FixedLenFeature([], tf.int64),
#         "face_valid": tf.io.FixedLenFeature([], tf.int64),
#         "face_x": tf.io.FixedLenFeature([], tf.int64),
#         "face_y": tf.io.FixedLenFeature([], tf.int64),
#         "face_w": tf.io.FixedLenFeature([], tf.int64),
#         "face_h": tf.io.FixedLenFeature([], tf.int64),
#         "leye_x": tf.io.FixedLenFeature([], tf.int64),
#         "leye_y": tf.io.FixedLenFeature([], tf.int64),
#         "leye_w": tf.io.FixedLenFeature([], tf.int64),
#         "leye_h": tf.io.FixedLenFeature([], tf.int64),
#         "reye_x": tf.io.FixedLenFeature([], tf.int64),
#         "reye_y": tf.io.FixedLenFeature([], tf.int64),
#         "reye_w": tf.io.FixedLenFeature([], tf.int64),
#         "reye_h": tf.io.FixedLenFeature([], tf.int64),
#         "dot_xcam": tf.io.FixedLenFeature([], tf.float32),
#         "dot_y_cam": tf.io.FixedLenFeature([], tf.float32),
#         "dot_x_pix": tf.io.FixedLenFeature([], tf.float32),
#         "dot_y_pix": tf.io.FixedLenFeature([], tf.float32),
#         "reye_x1": tf.io.FixedLenFeature([], tf.int64),
#         "reye_y1": tf.io.FixedLenFeature([], tf.int64),
#         "reye_x2": tf.io.FixedLenFeature([], tf.int64),
#         "reye_y2": tf.io.FixedLenFeature([], tf.int64),
#         "leye_x1": tf.io.FixedLenFeature([], tf.int64),
#         "leye_y1": tf.io.FixedLenFeature([], tf.int64),
#         "leye_x2": tf.io.FixedLenFeature([], tf.int64),
#         "leye_y2": tf.io.FixedLenFeature([], tf.int64),
#     }
#     example = tf.io.parse_single_example(example, feature_description)
#     example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
#     return example


# print('-> parse')


# def augmentation(image, training = True):
#     if training:
#         aug = tf.keras.Sequential([
#                 layers.Resizing(128+10, 128+10),
#                 layers.RandomCrop(128, 128, 256),
#                 layers.Rescaling(1./255),
#                 layers.Normalization(mean = (0.3741, 0.4076, 0.5425), variance = (0.0004, 0.0004, 0.0004))
#                 ])
        
#     else:
#         aug = tf.keras.Sequential([
#                 layers.Resizing(128+10, 128+10),
#                 layers.Rescaling(1./255),
#                 layers.Normalization(mean = (0.3741, 0.4076, 0.5425), variance = (0.0004, 0.0004, 0.0004))
#                 ])
    
#     image = aug(image)
    
#     return image

# def prepare_sample(features):
#     img_feat = features['image']

#     h = tf.shape(img_feat)[0]
#     w = tf.shape(img_feat)[1]

#     w = tf.cast(w, tf.int64)
#     h = tf.cast(h, tf.int64)

#     screen_w, screen_h = features['screen_w'], features['screen_h']

#     kps = [features['leye_x1']/w, features['leye_y1']/h, features['leye_x2']/w, features['leye_y2']/h,
#            features['reye_x1']/w, features['reye_y1']/h, features['reye_x2']/w, features['reye_y2']/h]
#     # kps has type float64

#     lx, ly, lw, lh = int(features['leye_x']), int(features['leye_y']), int(features['leye_w']), int(features['leye_h'])
#     rx, ry, rw, rh = int(features['reye_x']), int(features['reye_y']), int(features['reye_w']), int(features['reye_h'])

#     lx = tf.clip_by_value(lx, 0, int(w)-lw)
#     ly = tf.clip_by_value(ly, 0, int(h)-lh)

#     rx = tf.clip_by_value(rx, 0, int(w)-rw)
#     ry = tf.clip_by_value(ry, 0, int(h)-rh)

#     l_eye = tf.image.crop_to_bounding_box(img_feat, ly, lx, lh, lw)
#     r_eye = tf.image.crop_to_bounding_box(img_feat, ry, rx, rh, rw)

#     l_eye = tf.image.flip_left_right(l_eye)

#     l_eye = augmentation(l_eye)
#     r_eye = augmentation(r_eye)

#     y = [features['dot_xcam'], features['dot_y_cam']]
#     # y has type float32

#     return (l_eye, r_eye, kps), y

# def get_batched_dataset(filenames, batch_size):
#     option_no_order = tf.data.Options()
#     option_no_order.deterministic = False  # disable order, increase speed
    
#     dataset = (
#         tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
#         .with_options(option_no_order)
#         .map(parse_tfrecord_fn, num_parallel_calls=AUTO)
#         .map(prepare_sample, num_parallel_calls=AUTO)
#         .shuffle(batch_size*10)
#         .batch(batch_size)
#         .prefetch(buffer_size=AUTO)
#     )
    
#     dataset_len = sum(1 for _ in tf.data.TFRecordDataset(filenames))
#     print(f"No. of samples: {dataset_len}")
    
#     return dataset


# # train_dataset = get_batched_dataset(TRAINING_FILENAMES, BATCH_SIZE)
# valid_dataset = get_batched_dataset(VALID_FILENAMES, BATCH_SIZE)
# # test_dataset = get_batched_dataset(TEST_FILENAMES, BATCH_SIZE)

# class eye_model(layers.Layer):
#     def __init__(self, name='Eye-model'):
#         super(eye_model, self).__init__()

#         self.conv1 = layers.Conv2D(32, kernel_size=7, strides=2, padding='valid') 
#         self.conv2 = layers.Conv2D(64, kernel_size=5, strides=2, padding='valid')
#         self.conv3 = layers.Conv2D(128, kernel_size=3, strides=1, padding='valid')
#         self.bn1 = layers.BatchNormalization(axis = -1, momentum=0.9)
#         self.bn2 = layers.BatchNormalization(axis = -1, momentum=0.9)
#         self.bn3 = layers.BatchNormalization(axis = -1, momentum=0.9)
#         self.leakyrelu = layers.LeakyReLU(alpha=0.01)
#         self.avgpool = layers.AveragePooling2D(pool_size=2)
#         self.dropout = layers.Dropout(rate=0.02)

#     def call(self, input_image):
#         x = self.conv1(input_image)
#         x = self.bn1(x)
#         x = self.leakyrelu(x)
#         x = self.avgpool(x)
#         x = self.dropout(x) 

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.leakyrelu(x)
#         x = self.avgpool(x)
#         x = self.dropout(x)

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.leakyrelu(x)
#         x = self.avgpool(x)
#         x = self.dropout(x)

#         return x
    
#     def summary(self):
#         x = Input(shape=(128, 128, 3))
#         model = Model(inputs=[x], outputs=self.call(x))
#         return model.summary()



# class landmark_model(layers.Layer):
#     def __init__(self, name='Landmark-model'):
#         super(landmark_model, self).__init__()

#         self.dense1 = layers.Dense(128)
#         self.dense2 = layers.Dense(16)
#         self.dense3 = layers.Dense(16)
#         self.bn1 = layers.BatchNormalization(axis = -1,momentum=0.9)
#         self.bn2 = layers.BatchNormalization(axis = -1, momentum=0.9)
#         self.bn3 = layers.BatchNormalization(axis = -1, momentum=0.9)
#         self.relu = layers.ReLU()

#     def call(self, input_kps):
#         x = self.dense1(input_kps)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.dense2(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         x = self.dense3(x)
#         x = self.bn3(x)
#         x = self.relu(x)   

#         return x
    
#     def summary(self):
#         x = Input(shape=(8, ))
#         model = Model(inputs=[x], outputs=self.call(x))
#         return model.summary()

# class gazetrack_model(Model):
#     def __init__(self, name='Gazetrack-model'):
#         super(gazetrack_model, self).__init__()

#         self.eye_model = eye_model()
#         self.landmark_model = landmark_model()

#         self.dense1 = layers.Dense(8)
#         self.dense2 = layers.Dense(4)
#         self.dense3 = layers.Dense(2)

#         self.bn1 = layers.BatchNormalization(axis = -1, momentum=0.9)
#         self.bn2 = layers.BatchNormalization(axis = -1, momentum=0.9)
#         self.dropout = layers.Dropout(rate=0.12)
#         self.relu = layers.ReLU()


#     def call(self, l_r_lms):
#         # leftEye = l_r_lms['l_eye']
#         # rightEye = l_r_lms['r_eye']
#         # lms = l_r_lms['kps']
        
#         leftEye, rightEye, lms = l_r_lms
        
#         l_eye_feat = self.eye_model(leftEye)
#         r_eye_feat = self.eye_model(rightEye)
        
#         l_eye_feat = layers.Flatten()(l_eye_feat)
#         r_eye_feat = layers.Flatten()(r_eye_feat)

    
#         lm_feat = self.landmark_model(lms)
        
#         combined_feat = tf.concat((l_eye_feat, r_eye_feat, lm_feat),1)

#         x = self.dense1(combined_feat)
#         x = self.bn1(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         x = self.dense2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dense3(x)

#         return x
    
    
#     def summary(self):
#         input1 = Input(shape=(128,128,3))
#         input2 = Input(shape=(128,128,3))
#         input3 = Input(shape=(8, ))

#         model = Model(inputs=[input1, input2, input3], outputs=self.call([input1, input2, input3]))
#         return model.summary()
    

# print('-> model defined')

# lr = 0.016
# loss = tf.keras.losses.MeanSquaredError()
# metrics = [tf.keras.metrics.mean_squared_error]
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=0.9, beta_2=0.999, epsilon=1e-07)
# scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, mode='min')
# batch_size = 256

# print('-> training parameters set')

# model = gazetrack_model()
# print(model.summary())

# print('-> model summary printed')


# model.compile(optimizer=optimizer, loss=loss, metrics=metrics,)



# print('-> training next')


# model.fit(
#     x=valid_dataset,   
#     batch_size=batch_size,
#     epochs=1,  
#     verbose='auto',   #auto=1, 1=progress bar, 2=one line per epoch( maybe use 2 if running job)
#     callbacks=[scheduler],
#     validation_data=valid_dataset,
#     shuffle=True,    #probably will not work as our dataset is a tf.data object
#     initial_epoch=0,     #epoch at which to resume training
#     workers=1,
#     use_multiprocessing=False
# )

# print('-> model eval next')

# model.evaluate(
#     x=valid_dataset,
#     batch_size=batch_size,
#     verbose='auto'
# )


