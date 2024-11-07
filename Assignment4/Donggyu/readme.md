Answer of Question 1

Answer of Question 2

Answer of Question 3

l1_drop_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l1(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l1(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l1(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l1(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['l1_drop'] = compile_and_fit(l1_drop_model, "regularizers/l1_drop")

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
