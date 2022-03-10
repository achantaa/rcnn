import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import vgg16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


def get_iou(bb1, bb2):
    """Get the IOU for two intersecting bounding boxes"""
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def get_selective_search(annotations_path, images_path):
    """Selective Search for finding foreground and background images using IOU"""
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    train_images = []
    train_labels = []

    for file in tqdm(os.listdir(annotations_path)):
        if file.startswith('airplane'):
            name = file.split('.')[0] + '.jpg'
            img = cv2.imread(images_path + name)
            bboxes = pd.read_csv(os.path.join(annotations_path, file))
            actual_boxes = []
            for row in bboxes.iterrows():
                actual_boxes.append({'x1': int(row[1][0].split(' ')[0]),
                                     'y1': int(row[1][0].split(' ')[1]),
                                     'x2': int(row[1][0].split(' ')[2]),
                                     'y2': int(row[1][0].split(' ')[3])})

            selective_search.setBaseImage(img)
            selective_search.switchToSelectiveSearchFast()
            res = selective_search.process()
            output_img = img.copy()
            foreground_counter = 0
            background_counter = 0
            for i in range(min(len(res), 2000)):
                for actual_vals in actual_boxes:
                    x, y, w, h = res[i]
                    iou = get_iou(actual_vals, {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h})
                    if foreground_counter < 30:
                        if iou > 0.7:
                            f_sample = output_img[y: y + h, x: x + w]
                            train_images.append(cv2.resize(f_sample, (224, 224), interpolation=cv2.INTER_AREA))
                            train_labels.append(1)
                            foreground_counter += 1

                    if background_counter < 30:
                        if iou < 0.3:
                            b_sample = output_img[y: y + h, x: x + w]
                            train_images.append(cv2.resize(b_sample, (224, 224), interpolation=cv2.INTER_AREA))
                            train_labels.append(0)
                            background_counter += 1

                    if foreground_counter >= 30 and background_counter >= 30:
                        break

    return np.array(train_images), np.array(train_labels)


class RCNN(keras.Model):
    """Main RCNN class"""
    def __init__(self):
        super(RCNN, self).__init__()
        self.vgg = vgg16.VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
        for layers in self.vgg.layers[:15]:
            layers.trainable = False
        outs = self.vgg.layers[-2].output

        self.backend = keras.Model(self.vgg.input, outs, name='vgg_15')
        self.final = keras.layers.Dense(2, activation=keras.activations.softmax, name='output')

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        x = self.backend(x)
        x = self.final(x)
        return x

    def build_graph(self, input_shape):
        x = keras.Input(shape=input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y


def augment_images(X, y):
    data_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
    return data_gen.flow(x=X, y=y)


def create_dataset(annotations_path, images_path, train_img_path, train_label_path):
    train_images, train_labels = get_selective_search(annotations_path, images_path)
    np.save(train_img_path, train_images)
    np.save(train_label_path, train_labels)
    return train_images, train_labels


def load_dataset(train_img_path, train_label_path):
    return np.load(train_img_path), np.load(train_label_path)


def predict_output(model, images_path):
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    for file in tqdm(os.listdir(images_path)):
        if file.startswith('4'):
            img = cv2.imread(os.path.join(images_path, file))
            selective_search.setBaseImage(img)
            selective_search.switchToSelectiveSearchFast()
            res = selective_search.process()
            output_img = img.copy()

            for i in range(min(len(res), 2000)):
                x, y, w, h = res[i]
                region = output_img[y: y + h, x: x + w]
                region_resized = cv2.resize(region, (224, 224), interpolation=cv2.INTER_AREA)
                region_img = np.expand_dims(region_resized, axis=0)
                model_out = model.predict(region_img)

                if model_out[0][0] > 0.70:
                    cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), lineType=cv2.LINE_AA)

            plt.imshow(output_img)
            plt.show()


if __name__ == '__main__':
    train_img_path = '.\\data\\train_img.npy'
    train_label_path = '.\\data\\train_labels.npy'
    annotations_path = '.\\data\\airplanes\\Airplanes_Annotations\\'
    images_path = '.\\data\\airplanes\\Images\\'

    train_images, train_labels = create_dataset(annotations_path, images_path, train_img_path, train_label_path)
    # train_images, train_labels = load_dataset(train_img_path, train_label_path)

    binarizer = MyLabelBinarizer()
    Y = binarizer.fit_transform(train_labels)

    X_train, X_test, y_train, y_test = train_test_split(train_images, Y, test_size=0.10, shuffle=True)

    train_data = augment_images(X_train, y_train)
    test_data = augment_images(X_test, y_test)

    rcnn = RCNN()
    # model = rcnn.build_graph((224, 224, 3))
    # model.summary()

    optimizer = tf.optimizers.Adam(learning_rate=0.00005)
    rcnn.compile(optimizer=optimizer, loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])

    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto',
                          restore_best_weights=True)

    hist = rcnn.fit(train_data, steps_per_epoch=10, epochs=1000, validation_data=test_data,
                    validation_steps=2, callbacks=[early])

    rcnn.save('rcnn_model')
    # rcnn_loaded = keras.models.load_model('.\\rcnn_model')

    predict_output(rcnn, images_path)
