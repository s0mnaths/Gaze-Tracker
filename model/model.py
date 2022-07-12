import tensorflow as tf
from tensorflow.keras import Model, layers




class eye_model(Model):
  def __init__(self):
    super(eye_model, self).__init__(name='')

    self.conv1 = layers.Conv2D(32, kernel_size=7, strides=2, padding='valid')
    self.conv2 = layers.Conv2D(64, kernel_size=5, strides=2, padding='valid')
    self.conv3 = layers.Conv2D(128, kernel_size=3, strides=1, padding='valid')
    self.bn = layers.BatchNormalization(axis = 1, momentum=0.9)
    self.leakyrelu = layers.LeakyReLU(alpha=0.01) 
    self.avgpool = layers.AveragePooling2D(pool_size=2)
    self.dropout = layers.Dropout(rate=0.02)
    

  def call(self, input_tensor):
    x = self.conv1(input_tensor)
    x = self.bn(x)
    x = self.leakyrelu(x)
    x = self.avgpool(x)
    x = self.dropout(x)
    
    x = self.conv2(x)
    x = self.bn(x)
    x = self.leakyrelu(x)
    x = self.avgpool(x)
    x = self.dropout(x)
    
    x = self.conv3(x)
    x = self.bn(x)
    x = self.leakyrelu(x)
    x = self.avgpool(x)
    x = self.dropout(x)
    
    return x

class landmark_model(Model):
  def __init__(self):
    super(landmark_model, self).__init__(name='')

    self.dense1 = layers.Dense(128)
    self.dense2 = layers.Dense(16)
    self.dense3 = layers.Dense(16)
    self.bn = layers.BatchNormalization(momentum=0.9)
    self.relu = layers.ReLU()

  def call(self, input_tensor):
    x = self.dense1(input_tensor)
    x = self.bn(x)
    x = self.relu(x)
    
    x = self.dense2(x)
    x = self.bn(x)
    x = self.relu(x)
    
    x = self.dense3(x)
    x = self.bn(x)
    x = self.relu(x)   
    
    return x

class gazetrack_model(Model):
  def __init__(self):
    super(gazetrack_model, self).__init__(name='')

    self.eye_model = eye_model()
    self.lmModel = landmark_model()
    
    self.dense1 = layers.Dense(8)
    self.dense2 = layers.Dense(4)
    self.dense3 = layers.Dense(2)
    
    self.bn = layers.BatchNormalization(momentum=0.9)
    self.dropout = layers.Dropout(rate=0.12)
    self.relu = layers.ReLU()

    

  def call(self, leftEye, rightEye, lms):
    l_eye_feat = tf.reshape(self.eye_model(leftEye), (3, 128*128))
    r_eye_feat = tf.reshape(self.eye_model(rightEye), (3, 128*128))
    
    lm_feat = self.lmModel(lms)
    
    combined_feat = tf.concat((l_eye_feat, r_eye_feat, lm_feat),1)
    
    x = self.dense1(combined_feat)
    x = self.bn(x)
    x = self.dropout(x)
    x = self.relu(x)
    
    x = self.dense2(x)
    x = self.bn(x)
    x = self.relu(x)
    
    x = self.dense3(x)
    
    return x

model_predict = Model.predict()
model_predict_batch = Model.predict_on_batch()

model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['accuracy'])




lr =  0.016
# loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=0.9, beta_2=0.999, epsilon=1e-07)

scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, mode='min')

