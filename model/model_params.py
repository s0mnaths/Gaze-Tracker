import tensorflow as tf

lr = 0.016
loss = tf.keras.losses.MeanSquaredError()
metrics = [tf.keras.metrics.mean_squared_error, tf.keras.metrics.Accuracy]
optimizer = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=0.9, beta_2=0.999, epsilon=1e-07)
scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, mode='min')