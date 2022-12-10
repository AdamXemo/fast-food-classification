from data import xemo
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import random


size = 75

train_dir = "/home/adam/Datasets/Fast Food Classification V2/Train"
test_dir = "/home/adam/Datasets/Fast Food Classification V2/Test"

categories = ["Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries", "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"]

(x_train, y_train), (x_test, y_test) = xemo.format_data(train_dir, test_dir, categories, size, gray=True)

n = random.randrange(50, 14950)

for i in range(10):
    plt.imshow(x_train[i+n], cmap='gray')
    plt.title(categories[y_train[i+n][0]])
    plt.show()

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(size,size, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), activation='relu'))          
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.33))

model.add(Conv2D(32, (3,3), activation='relu'))          
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.33))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(10, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32)

model.save('image_to_data/model.h5')

model.evaluate(x_test, y_test)

