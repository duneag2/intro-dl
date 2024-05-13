
Q2 learning rate scheduler는 주로 모델의 optimal point를 찾기 위해서 사용합니다. 학습률이 너무 크면 비용이 최소가 되는 점에 도달하지 못할 수 있기 때문입니다. 

Q3 Dropout 도 사용해보세요~! 

EX) 
answer_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l1(0.001),
                 activation = 'elu', input_shape = (FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l1(0.001),
                 activation = 'elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l1(0.001),
                 activation = 'elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l1(0.001),
                 activation = 'elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])
