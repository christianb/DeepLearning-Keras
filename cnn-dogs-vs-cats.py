from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from plot import Plot

_EPOCHS_TRAIN = 30  # 30
_BATCH_SIZE = 20

_DATA_DIR = 'data/'
_PARTIAL_DATASET_DIR = os.path.join(_DATA_DIR, 'dogs-vs-cats-partial')
_TRAIN_DIR = os.path.join(_PARTIAL_DATASET_DIR, 'train')
_VALIDATION_DIR = os.path.join(_PARTIAL_DATASET_DIR, 'validation')
_MODEL_SAVE_FILE_PATH = os.path.join(_DATA_DIR, 'dogs_and_cats_small_1.h5')
_TRAIN_CATS_DIR = os.path.join(_TRAIN_DIR, "cats")


def __prepare_data():
    original_dataset_dir = os.path.join(_DATA_DIR, 'dogs-vs-cats-complet')

    os.mkdir(_PARTIAL_DATASET_DIR)
    os.mkdir(_TRAIN_DIR)
    os.mkdir(_VALIDATION_DIR)

    test_dir = os.path.join(_PARTIAL_DATASET_DIR, "test")
    os.mkdir(test_dir)
    os.mkdir(_TRAIN_CATS_DIR)

    # directory for train dog images
    train_dogs_dir = os.path.join(_TRAIN_DIR, "dogs")
    os.mkdir(train_dogs_dir)

    # directory validation cat images
    validation_cats_dir = os.path.join(_VALIDATION_DIR, "cats")
    os.mkdir(validation_cats_dir)

    # directory validation dog images
    validation_dogs_dir = os.path.join(_VALIDATION_DIR, "dogs")
    os.mkdir(validation_dogs_dir)

    # directory test cat images
    test_cats_dir = os.path.join(test_dir, "cats")
    os.mkdir(test_cats_dir)

    # directory test dog images
    test_dogs_dir = os.path.join(test_dir, "dogs")
    os.mkdir(test_dogs_dir)

    # copy the first 1000 cat images to train dir
    __copy_with_range('cat', 0, 1000, original_dataset_dir, _TRAIN_CATS_DIR)

    # copy the next 500 cat images to validation dir
    __copy_with_range('cat', 1000, 1500, original_dataset_dir, validation_cats_dir)

    # copy the next 500 cat images to test dir
    __copy_with_range('cat', 1500, 2000, original_dataset_dir, test_cats_dir)

    # copy the first 1000 dog images to train dir
    __copy_with_range('dog', 0, 1000, original_dataset_dir, train_dogs_dir)

    # copy the next 500 dog images to validation dir
    __copy_with_range('dog', 1000, 1500, original_dataset_dir, validation_dogs_dir)

    # copy the next 500 dog images to test dir
    __copy_with_range('dog', 1500, 2000, original_dataset_dir, test_dogs_dir)

    print('number cat train images: ', len(os.listdir(_TRAIN_CATS_DIR)))
    print('number dog train images: ', len(os.listdir(train_dogs_dir)))

    print('number cat validation images: ', len(os.listdir(validation_cats_dir)))
    print('number dog validation images: ', len(os.listdir(validation_dogs_dir)))

    print('number cat test images: ', len(os.listdir(test_cats_dir)))
    print('number dog test images: ', len(os.listdir(test_dogs_dir)))


def __copy_with_range(type, min, max, src_dir, dst_dir):
    fnames = ['{}.{}.jpg'.format(type, i) for i in range(min, max)]
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        shutil.copyfile(src, dst)


def __build_model():
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
    return model


def __data_augmentation_example():
    datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

    import matplotlib.pyplot as plt
    from keras.preprocessing import image

    fnames = [os.path.join(_TRAIN_CATS_DIR, fname) for fname in os.listdir(_TRAIN_CATS_DIR)]
    img_path = fnames[3]
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        plt.imshow(image.array_to_img(batch[0]))
        i += 1
        plt.savefig('outputs/cnn-dogs-vs-cats/data_augmentation_example_{}.png'.format(i))
        if i % 4 == 0:
            break


if __name__ == '__main__':
    # __prepare_data()

    # rescale with a factor of 1/255
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(_TRAIN_DIR,
                                                        target_size=(150, 150),
                                                        batch_size=_BATCH_SIZE,
                                                        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(_VALIDATION_DIR,
                                                                  target_size=(150, 150),
                                                                  batch_size=_BATCH_SIZE,
                                                                  class_mode='binary')

    # print shape of train_generator batches
    # for data_batch, labels_batch in train_generator:
    #     print("Shape data_batch: ", data_batch.shape)
    #     print("Shape labels_batch: ", labels_batch.shape)
    #     break

    model = __build_model()

    # one step contains 20 batches, to train over all 2000 samples we need 100 steps per epoch
    # one validation step contains 20 batches, to validate over all 1000 samples we need 50 steps per validation
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=_EPOCHS_TRAIN,
                                  validation_data=validation_generator,
                                  validation_steps=50)

    model.save(_MODEL_SAVE_FILE_PATH)

    plot = Plot(history, 'CNN Dogs vs. Cats', "cnn-dogs-vs-cats")
    plot.plot()

    # __data_augmentation_example()