import tensorflow as tf
from tensorflow.keras import Input, Model, layers


class eye_model(layers.Layer):
    def __init__(self, name='Eye-model'):
        super(eye_model, self).__init__()

        self.conv1 = layers.Conv2D(32, kernel_size=7, strides=2, padding='valid') 
        self.conv2 = layers.Conv2D(64, kernel_size=5, strides=2, padding='valid')
        self.conv3 = layers.Conv2D(128, kernel_size=3, strides=1, padding='valid')
        self.bn1 = layers.BatchNormalization(axis = -1, momentum=0.9)
        self.bn2 = layers.BatchNormalization(axis = -1, momentum=0.9)
        self.bn3 = layers.BatchNormalization(axis = -1, momentum=0.9)
        self.leakyrelu = layers.LeakyReLU(alpha=0.01)
        self.avgpool = layers.AveragePooling2D(pool_size=2)
        self.dropout = layers.Dropout(rate=0.02)

    def call(self, input_image):
        x = self.conv1(input_image)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.avgpool(x)
        x = self.dropout(x) 

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        return x
    



class landmark_model(layers.Layer):
    def __init__(self, name='Landmark-model'):
        super(landmark_model, self).__init__()

        self.dense1 = layers.Dense(128)
        self.dense2 = layers.Dense(16)
        self.dense3 = layers.Dense(16)
        self.bn1 = layers.BatchNormalization(axis = -1,momentum=0.9)
        self.bn2 = layers.BatchNormalization(axis = -1, momentum=0.9)
        self.bn3 = layers.BatchNormalization(axis = -1, momentum=0.9)
        self.relu = layers.ReLU()

    def call(self, input_kps):
        x = self.dense1(input_kps)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dense2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.dense3(x)
        x = self.bn3(x)
        x = self.relu(x)   

        return x
    

class gazetrack_model(layers.Layer):
    def __init__(self, name='Gazetrack-model'):
        super(gazetrack_model, self).__init__()

        self.eye_model = eye_model()
        self.landmark_model = landmark_model()

        self.dense1 = layers.Dense(8)
        self.dense2 = layers.Dense(4)
        self.dense3 = layers.Dense(2)

        self.bn1 = layers.BatchNormalization(axis = -1, momentum=0.9)
        self.bn2 = layers.BatchNormalization(axis = -1, momentum=0.9)
        self.dropout = layers.Dropout(rate=0.12)
        self.relu = layers.ReLU()


    def call(self, l_r_lms):
        
        leftEye, rightEye, lms = l_r_lms
        
        l_eye_feat = self.eye_model(leftEye)
        r_eye_feat = self.eye_model(rightEye)
        
        l_eye_feat = layers.Flatten()(l_eye_feat)
        r_eye_feat = layers.Flatten()(r_eye_feat)

    
        lm_feat = self.landmark_model(lms)
        
        combined_feat = tf.concat((l_eye_feat, r_eye_feat, lm_feat),1)

        x = self.dense1(combined_feat)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)

        return x