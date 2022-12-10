import numpy as np
import random
import os 
import cv2
import time


class xemo:
    @staticmethod
    def format_data(train_directory, test_directory, categories, img_size, gray=False):

        if gray==True:
            color = cv2.IMREAD_GRAYSCALE
        else:
            color = cv2.IMREAD_COLOR
        
        print()

        training_data = []
        for i, category in enumerate(categories):
            path = os.path.join(train_directory, category)
            train_labels = categories.index(category)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, img), color)
                img_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([img_array, train_labels])
            for r in range(10):
                time.sleep(0.08)
                print(f"Train images {i*10+r}%\r", end='')
        

        print(f"Train images formating completed!\r")

        testing_data = []
        for i, category in enumerate(categories):
            path = os.path.join(test_directory, category)
            test_labels = categories.index(category)
            for img in os.listdir(path):
                img_array2 = cv2.imread(os.path.join(path, img), color)
                img_array2 = cv2.resize(img_array2, (img_size, img_size))
                testing_data.append([img_array2, test_labels])
            for r in range(10):
                time.sleep(0.08)
                print(f"Test images {i*10+r}%\r", end='')                

        print(f"Test images formating completed!\r")

        random.shuffle(training_data)
        random.shuffle(testing_data)


        x_train = np.array([i[0] for i in training_data])
        y_train = np.array([i[1] for i in training_data])
        y_train.resize(15000,1)

        x_test = np.array([i[0] for i in testing_data])
        y_test = np.array([i[1] for i in testing_data])
        y_test.resize(1500, 1)

        x_train = x_train/255
        x_test = x_test/255


        return (x_train, y_train), (x_test, y_test)


'''
train_dir = "/home/adam/Datasets/Fast Food Classification V2/Train"
test_dir = "/home/adam/Datasets/Fast Food Classification V2/Test"

size = 75

categories = ["Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries", "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"]

(x_train, y_train), (x_test, y_test) = xemo.format_data(train_dir, test_dir, categories, size, gray=True)
'''
