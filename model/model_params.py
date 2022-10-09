import tensorflow as tf

lr = 0.016
loss = tf.keras.losses.MeanSquaredError() 
metrics = [tf.keras.metrics.RootMeanSquaredError()]

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=8000,
    decay_rate=0.64,
    staircase=True)




optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,beta_1=0.9, beta_2=0.999, epsilon=1e-07)
# scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, mode='min')
# schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, 8000, 0.64, staircase=True)
# scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)