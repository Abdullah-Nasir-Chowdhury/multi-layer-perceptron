import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs

classes = 4
m = 100
centers = [[-5,2], [-2,-2], [1,2], [5,-2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, 
                              centers=centers, 
                              cluster_std=std, 
                              random_state=30)

print(f"Unique classes {np.unique(y_train)}")
print(f"Class representation {y_train[:10]}")
print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")


tf.random.set_seed(1234)
model = Sequential([
    Dense(units=25, activation='relu', name='Layer1'),
    Dense(units=15, activation='relu', name='Layer2'),
    Dense(units=10, activation='linear', name='Layer3')
])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
model.fit(X_train, y_train, epochs=200)
logits = model(X_train)
f_x = tf.nn.softmax(logits)


l1 = model.get_layer('Layer1')
l2 = model.get_layer('Layer2')
l3 = model.get_layer('Layer3')

W1, b1 = l1.get_weights()
W2, b2 = l2.get_weights()
W3, b3 = l3.get_weights()





