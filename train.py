import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# th oznacava da ce konvoluciono jezgro biti oblika (depth, input_depth, rows, cols)
K.set_image_dim_ordering('th')

img_width, img_height = 150, 150

# postavljamo parametre
train_data_dir = 'data/train'
validation_data_dir = 'data/validate'
nb_train_samples = 100
nb_validation_samples = 25
nb_epoch = 1000


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()

# Konvolucija i sažimanje 1 
model.add(Conv2D(32, 3, 3, input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Konvolucija i sažimanje 2
model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Konvolucija i sažimanje 3
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Transformacija 3D slike u 1D vektor 
model.add(Flatten())
model.add(Dense(64)) # viseslojni perceptron
model.add(Activation('relu'))
model.add(Dropout(0.5)) # deaktiviramo polovinu neurona 
model.add(Dense(5))  # broj klasa
model.add(Activation('softmax'))

model.summary()

rmsprop = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-8, decay=0.0)

# Konfigurisanje modela
model.compile(
    loss='categorical_crossentropy',
    optimizer=rmsprop,
    metrics=['accuracy'])
batch_size = 32

# slucajne transformacije
train_datagen = ImageDataGenerator( 
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# treniranje mreze
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    nb_epoch=nb_epoch,
    # validation_data=validation_generator,
    nb_val_samples=nb_validation_samples//batch_size)

# pamtimo model i tezine modela
model.save_weights('first_try.h5')