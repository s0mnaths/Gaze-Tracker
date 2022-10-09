import tensorflow as tf
import json
import os 


# PATH_MAIN = './' #path where the gazetrack data is unzipped 
# TFREC_PARENT_PATH = '/home/s0mnaths/projects/def-skrishna/s0mnaths/datasets/mit_split_tfrec/' #path parent path to save the tfrec folder



def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(image, path, example):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(path),
        "device": bytes_feature(example["device"]),
        "screen_h": int64_feature(example["screen_h"]),
        "screen_w": int64_feature(example["screen_w"]),
        "face_valid": int64_feature(example["face_valid"]),
        "face_x": int64_feature(example["face_x"]),
        "face_y": int64_feature(example["face_y"]),
        "face_w": int64_feature(example["face_w"]),
        "face_h": int64_feature(example["face_h"]),
        "leye_x": int64_feature(example["leye_x"]),
        "leye_y": int64_feature(example["leye_y"]),
        "leye_w": int64_feature(example["leye_w"]),
        "leye_h": int64_feature(example["leye_h"]),
        "reye_x": int64_feature(example["reye_x"]),
        "reye_y": int64_feature(example["reye_y"]),
        "reye_w": int64_feature(example["reye_w"]),
        "reye_h": int64_feature(example["reye_h"]),
        "dot_xcam": float_feature(example["dot_xcam"]),
        "dot_y_cam": float_feature(example["dot_y_cam"]),
        "dot_x_pix": float_feature(example["dot_x_pix"]),
        "dot_y_pix": float_feature(example["dot_y_pix"]),
        "reye_x1": int64_feature(example["reye_x1"]),
        "reye_y1": int64_feature(example["reye_y1"]),
        "reye_x2": int64_feature(example["reye_x2"]),
        "reye_y2": int64_feature(example["reye_y2"]),
        "leye_x1": int64_feature(example["leye_x1"]),
        "leye_y1": int64_feature(example["leye_y1"]),
        "leye_x2": int64_feature(example["leye_x2"]),
        "leye_y2": int64_feature(example["leye_y2"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))



def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "device": tf.io.FixedLenFeature([], tf.string),
        "screen_h": tf.io.FixedLenFeature([], tf.int64),
        "screen_w": tf.io.FixedLenFeature([], tf.int64),
        "face_valid": tf.io.FixedLenFeature([], tf.int64),
        "face_x": tf.io.FixedLenFeature([], tf.int64),
        "face_y": tf.io.FixedLenFeature([], tf.int64),
        "face_w": tf.io.FixedLenFeature([], tf.int64),
        "face_h": tf.io.FixedLenFeature([], tf.int64),
        "leye_x": tf.io.FixedLenFeature([], tf.int64),
        "leye_y": tf.io.FixedLenFeature([], tf.int64),
        "leye_w": tf.io.FixedLenFeature([], tf.int64),
        "leye_h": tf.io.FixedLenFeature([], tf.int64),
        "reye_x": tf.io.FixedLenFeature([], tf.int64),
        "reye_y": tf.io.FixedLenFeature([], tf.int64),
        "reye_w": tf.io.FixedLenFeature([], tf.int64),
        "reye_h": tf.io.FixedLenFeature([], tf.int64),
        "dot_xcam": tf.io.FixedLenFeature([], tf.float32),
        "dot_y_cam": tf.io.FixedLenFeature([], tf.float32),
        "dot_x_pix": tf.io.FixedLenFeature([], tf.float32),
        "dot_y_pix": tf.io.FixedLenFeature([], tf.float32),
        "reye_x1": tf.io.FixedLenFeature([], tf.int64),
        "reye_y1": tf.io.FixedLenFeature([], tf.int64),
        "reye_x2": tf.io.FixedLenFeature([], tf.int64),
        "reye_y2": tf.io.FixedLenFeature([], tf.int64),
        "leye_x1": tf.io.FixedLenFeature([], tf.int64),
        "leye_y1": tf.io.FixedLenFeature([], tf.int64),
        "leye_x2": tf.io.FixedLenFeature([], tf.int64),
        "leye_y2": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example




# path_distinct = []
# for x in sorted(os.listdir(PATH_MAIN)):
#     path_distinct.append(x)
    
# path_test = PATH_MAIN + path_distinct[1]
# path_train = PATH_MAIN + path_distinct[2]
# path_val = PATH_MAIN + path_distinct[3]

# paths_diff = [path_test,path_train,path_val]


# tfrec_paths = [TFREC_PARENT_PATH + 'test.tfrec', TFREC_PARENT_PATH + 'train.tfrec', TFREC_PARENT_PATH + 'val.tfrec']



# for i,x in enumerate(paths_diff):
#     x_img = x+ '/images'
#     x_json = x+'/meta'
#     with tf.io.TFRecordWriter(tfrec_paths[i]) as writer:
#         for y in sorted(os.listdir(x_img)):
#             temp_path = x_img+'/'+y
#             image_path = temp_path
#             temp_path = temp_path.split('/')
#             temp_path = temp_path[-1].split('.j')
#             img_id = temp_path[0]
#             json_path = x_json + '/' + img_id + '.json'
#             image = tf.io.decode_jpeg(tf.io.read_file(image_path))
#             json_file = json.load(open(json_path))
#             example = create_example(image, image_path, json_file)
#             writer.write(example.SerializeToString())