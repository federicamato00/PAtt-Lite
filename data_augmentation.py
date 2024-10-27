

from sklearn.utils import shuffle
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
import h5py



def load_data(path_prefix, dataset_name):
    name_path = dataset_name + '_numClasses7'
    path = os.path.join(path_prefix, name_path)
    file_path = os.path.join(path, dataset_name.lower() + '_data_augmentation.h5')

    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []

    with h5py.File(file_path, 'r') as f:
        X_train = np.array(f['X_train'])
        y_train = np.array(f['y_train'])
        X_val = np.array(f['X_val'])
        y_val = np.array(f['y_val'])
        X_test = np.array(f['X_test'])
        y_test = np.array(f['y_test'])

    return {'train': X_train, 'val': X_val, 'test': X_test}, {'train': y_train, 'val': y_val, 'test': y_test}


def get_unique_directory(base_path, dataset_name, dataset_name_2=None):
    dataset_dir = os.path.join(base_path, dataset_name, dataset_name_2) if dataset_name_2 else os.path.join(base_path, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    return dataset_dir


def augment_images(images, labels, datagen, target_count):
    augmented_images = []
    augmented_labels = []
    for image, label in zip(images, labels):
        image = image.reshape((1,) + image.shape)  # Reshape per il generatore
        i = 0
        for batch in datagen.flow(image, batch_size=1):
            augmented_images.append(batch[0])
            augmented_labels.append(label)
            i += 1
            if i >= target_count:
                break
        if len(augmented_images) >= target_count:
            break
    return np.array(augmented_images), np.array(augmented_labels)

def data_augmentation(use_augmentation, additional_augmentation, additional_images_per_class, path_distribution, classNames, X_train, y_train, X_val, y_val, X_test, y_test):
    if use_augmentation:

        # Configura il generatore di immagini
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.1,
            brightness_range=[0.8, 1.2],
            shear_range=0.1,
            zoom_range=0.2,
            channel_shift_range=0.0,
            fill_mode='nearest',
            cval=0.0,
            horizontal_flip=False,
            vertical_flip=False,
            rescale=None,
            validation_split=0.0,
            interpolation_order=1,
        )

        datagen.fit(X_train)

        def augment_and_balance(X, y, augmentations, additional_images_per_class):
            X_augmented, y_augmented = [], []
            unique_classes, class_counts = np.unique(y, return_counts=True)
            max_count = np.max(class_counts)
            for class_idx in unique_classes:
                class_images = X[y == class_idx]
                class_labels = y[y == class_idx]
                target_count = max_count - len(class_images)
                if target_count > 0:
                    augmented_images, augmented_labels = augment_images(class_images, class_labels, augmentations, target_count)
                    X_augmented.append(augmented_images)
                    y_augmented.append(augmented_labels)

            X_augmented = np.concatenate(X_augmented, axis=0)
            y_augmented = np.concatenate(y_augmented, axis=0)

            X = np.concatenate((X, X_augmented), axis=0)
            y = np.concatenate((y, y_augmented), axis=0)

            X_augmented, y_augmented = [], []
            for class_idx in unique_classes:
                class_images = X[y == class_idx]
                class_labels = y[y == class_idx]
                target_count = additional_images_per_class
                augmented_images, augmented_labels = augment_images(class_images, class_labels, augmentations, target_count)
                X_augmented.append(augmented_images)
                y_augmented.append(augmented_labels)


            X_augmented = np.concatenate(X_augmented, axis=0)
            y_augmented = np.concatenate(y_augmented, axis=0)

            X = np.concatenate((X, X_augmented), axis=0)
            y = np.concatenate((y, y_augmented), axis=0)

            return X, y


        X_train, y_train = augment_and_balance(X_train, y_train, datagen, additional_images_per_class)
        X_val, y_val = augment_and_balance(X_val, y_val, datagen, additional_images_per_class)

        plt.figure(figsize=(10, 5))
        plt.bar(np.unique(y_train), np.bincount(y_train))
        plt.title('Distribuzione delle classi dopo data augmentation (train)')
        plt.xlabel('Classi')
        plt.ylabel('Numero di campioni')
        plt.xticks(ticks=np.unique(y_train), labels=[classNames[i] for i in np.unique(y_train)])
        plt.savefig(os.path.join(path_distribution,dataset_name+'_numClasses7', 'dopo_augmentation_train.png'))

        plt.figure(figsize=(10, 5))
        plt.bar(np.unique(y_val), np.bincount(y_val))
        plt.title('Distribuzione delle classi dopo data augmentation (val)')
        plt.xlabel('Classi')
        plt.ylabel('Numero di campioni')
        plt.xticks(ticks=np.unique(y_val), labels=[classNames[i] for i in np.unique(y_val)])
        plt.savefig(os.path.join(path_distribution,dataset_name+'_numClasses7', 'dopo_augmentation_val.png'))

        X_train,y_train = shuffle(X_train, y_train, random_state=42)
        X_val,y_val = shuffle(X_val, y_val, random_state=42)

        # Visualizza alcune immagini augmentate
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(X_train[i].astype('uint8'))
            plt.title(f'Classe: {classNames[y_train[i]]}')
            plt.axis('off')
        plt.suptitle('Esempi di immagini dopo data augmentation (train)')
        plt.show()



    def more_augmentation (X,y,path_distribution,classNames):
            '''prova ad aumentare il numero di campioni per classe'''
            '''salvato in boss_data_augmentation_2.h5'''
            # Aumenta ulteriormente il numero di dati per ogni classe
            unique_classes, class_counts = np.unique(y, return_counts=True)
            X_augmented, y_augmented = [], []
            for class_idx in unique_classes:
                class_images = X[y == class_idx]
                class_labels = y[y == class_idx]
                target_count = additional_images_per_class
                augmented_images, augmented_labels = augment_images(class_images, class_labels, datagen, target_count)
                X_augmented.append(augmented_images)
                y_augmented.append(augmented_labels)

            X_augmented = np.concatenate(X_augmented, axis=0)
            y_augmented = np.concatenate(y_augmented, axis=0)

            X = np.concatenate((X, X_augmented), axis=0)
            y = np.concatenate((y, y_augmented), axis=0)
            print("Ulteriore aumento del numero di campioni per classe completato.")
            # Visualizza la distribuzione delle classi dopo ulteriore aumento
            plt.figure(figsize=(10, 5))
            plt.bar(np.unique(y), np.bincount(y))
            plt.title('Distribuzione delle classi dopo ulteriore aumento')
            plt.xlabel('Classi')
            plt.ylabel('Numero di campioni')
            plt.xticks(ticks=np.unique(y), labels=[classNames[i] for i in np.unique(y)])
            plt.savefig(os.path.join(path_distribution,dataset_name+'_numClasses7', 'dopo_ulteriore_aumento_2.png'))

            return X,y

    if additional_augmentation == True:
            X_train, y_train = more_augmentation(X_train,y_train,path_distribution,classNames)
            X_val, y_val = more_augmentation(X_val,y_val,path_distribution,classNames)

    return {'train': X_train, 'val': X_val, 'test': X_test}, {'train': y_train, 'val': y_val, 'test': y_test}



dataset_name='CK+'

print("Loading data...")
path_prefix = os.path.join('datasets', 'processed')
X, y = load_data(path_prefix,dataset_name)
if 'CK+' in dataset_name:
    file_output = 'ckplus_data_augmentation'
elif 'RAFDB' in dataset_name:
    file_output = 'rafdb'
elif 'FERP' in dataset_name:
    file_output = 'ferp'
elif 'JAFFE' in dataset_name:
    file_output = 'jaffe'
elif 'Bosphorus' in dataset_name:
    file_output = 'bosphorus_data_augmentation'
else:
    file_output = 'dataset'

file_output = file_output + '.h5'

IMG_SIZE = 224 if 'RAFDB' in dataset_name else 120

if 'RAFDB' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'FERP' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'JAFFE' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
elif 'Bosphorus' in dataset_name:
    classNames = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise','neutral']
elif 'CK+' in dataset_name:
    classNames = ['neutral', 'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'] #7 Classi
else:
    classNames = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']


X,y = data_augmentation(True, False, 100, 'distribution_data_augmentation',classNames, X['train'], y['train'], X['val'], y['val'], X['test'], y['test'])

plt.imshow(X['train'][0])
plt.show()
new_folder = dataset_name + '_numClasses7'
path_new = os.path.join('datasets', 'data_augmentation', new_folder)
file_path_save = os.path.join(path_new,file_output)
with h5py.File(file_path_save, 'w') as dataset:
    for split in X.keys():
        dataset.create_dataset(f'X_{split}', data=X[split])
        dataset.create_dataset(f'y_{split}', data=y[split])

print('X shape:', X['train'].shape)
print('y shape:', y['train'].shape)
print('X shape:', X['val'].shape)
print('y shape:', y['val'].shape)
del X, y

print(f"Dataset salvato in {file_path_save}")


