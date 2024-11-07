Answer of Question 1

optimizer = optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999), weight_decay = 0.01)

optim.Adam : Adam을 optimizer로써 사용한다는 코드이다.
model.parameteres() : 설정된 모델의 파라미터를 입력받겠다는 코드이다.
lr : learning rate를 의미하며 기본값은 0.001이다. 학습을 한 번 할 때 얼마나 진행할지를 표시하며 크기가 작으면 속도가 느리지만 정확하고, 크기가 크면 속도가 빠르지만 정확성이 비교적 떨어진다.
betas : 두 개의 인자를 필요로 한다. 첫 번째 값은 가중치를 의미하고, 두 번째 값은 lr을 얼마나 조절할지를 의미한다. 기본값은 (0.9, 0.999)이다.
weight_decay : 가중치를 얼마나 감소시킬지를 나타내는 숫자이며 기본값은 0이다.

Answer of Question 2

model = Net().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : 0.95 ** epoch)

for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval = 200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, test_loss, test_accuracy))

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
