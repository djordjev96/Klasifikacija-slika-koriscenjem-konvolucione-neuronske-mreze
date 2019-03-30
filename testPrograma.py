import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt


img_width, img_height = 128, 128
def create_model():
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # Konvolucija i sažimanje 1
    model.add(Conv2D(32, (3, 3), input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Konvolucija i sažimanje 2
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Konvolucija i sažimanje 3
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Transformacija 3D slike u 1D vektor
    model.add(Flatten())
    model.add(Dense(64))  # viseslojni perceptron
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # deaktiviramo polovinu neurona
    model.add(Dense(5))  # broj klasa
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    model = create_model()
    model.load_weights("./first_try.h5")
    rmsprop = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    img = image.load_img('testSlika.jpg', target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict_classes(images)
    classes2 = model.predict_proba(images)

    food = ""
    if classes[0] == 0:
        food = "Hamburger"
    if classes[0] == 1:
        food = "Hot dog"
    if classes[0] == 2:
        food = "Ice cream"
    if classes[0] == 3:
        food = "Pancakes"
    if classes[0] == 4:
        food = "Pizza"

    objects = ('Hamburger', 'Hot dog', 'Ice cream', 'Pancakes', 'Pizza')
    y_pos = np.arange(len(objects))
    performance = np.squeeze(np.asarray(classes2))

    plt.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    plt.xlabel('Usage')
    plt.title("Predicted: " + food)
    plt.show()
