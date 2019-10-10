from keras import layers
from keras import models
from keras import optimizers
import os
import shutil

_EPOCHS_TRAIN = 1  # 5
_BATCH_SIZE = 64


def __prepare_data():
    original_dataset_dir = 'data/dogs-vs-cats-complet/'
    base_dir = 'data/dogs-vs-cats-partial/'
    os.mkdir(base_dir)

    # directories for train, validation and test data
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)

    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)

    test_dir = os.path.join(base_dir, "test")
    os.mkdir(test_dir)

    # directory for train cat images
    train_cats_dir = os.path.join(train_dir, "cats")
    os.mkdir(train_cats_dir)

    # directory for train dog images
    train_dogs_dir = os.path.join(train_dir, "dogs")
    os.mkdir(train_dogs_dir)

    # directory validation cat images
    validation_cats_dir = os.path.join(validation_dir, "cats")
    os.mkdir(validation_cats_dir)

    # directory validation dog images
    validation_dogs_dir = os.path.join(validation_dir, "dogs")
    os.mkdir(validation_dogs_dir)

    # directory test cat images
    test_cats_dir = os.path.join(test_dir, "cats")
    os.mkdir(test_cats_dir)

    # directory test dog images
    test_dogs_dir = os.path.join(test_dir, "dogs")
    os.mkdir(test_dogs_dir)

    # copy the first 1000 cat images to train dir
    __copy_with_range('cat', 0, 1000, original_dataset_dir, train_cats_dir)

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

    print('number cat train images: ', len(os.listdir(train_cats_dir)))
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

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
    return model


if __name__ == '__main__':
    # __prepare_data()

    model = __build_model()
    # model.fit(train_images, train_labels, epochs=_EPOCHS_TRAIN, batch_size=_BATCH_SIZE)
