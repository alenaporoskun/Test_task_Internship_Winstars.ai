# Importing Libraries

# Provides OS realted functions
import os 

# For Image Manipulation
import matplotlib
import matplotlib.pyplot as plt

import skimage
from skimage.io import imread
from skimage.morphology import label

# For Data Manipulation and Analysis
import pandas as pd
import numpy as np

# Deep Learning Library
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model

# Other utilities
import random
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Directories Paths
train_dir = 'train_v2/'
test_dir = 'test_v2/'

def main():
    print_versions_libraries()

    # Directories Paths
    train_dir = 'train_v2/'
    test_dir = 'test_v2/'

    # Reading Training Data
    train_df = pd.read_csv('train_ship_segmentations_v2.csv')

    # Removing Bug Images
    train_df = train_df[train_df['ImageId'] != '6384c3e78.jpg']

    # Removing 10000 non-ship Images
    def area_isnull(x):
        if x == x:
            return 0
        else:
            return 1
        
    train_df['isnan'] = train_df['EncodedPixels'].apply(area_isnull)
    train_df = train_df.sort_values('isnan', ascending=False)
    train_df = train_df.iloc[100000:]

    # Now we will set the class for ship area

    # Calculate the area for each ship using the RLE-encoded pixels and create a new 'area' column
    train_df['area'] = train_df["EncodedPixels"].apply(calc_area_for_rle)

    # Select rows where the ship area is greater than 0 (indicating the presence of a ship)
    train_df_isship = train_df[train_df['area'] > 0]

    # Select rows where the ship area is less than 10
    train_df_smallarea = train_df_isship['area'][train_df_isship['area'] < 10]

    # Group the DataFrame by 'ImageId' and calculate the sum of ship areas for each image
    train_gp = train_df.groupby('ImageId').sum()
    train_gp = train_gp.reset_index()

    # Apply a function to classify each image based on the total ship area
    train_gp['class'] = train_gp['area'].apply(calc_class)

    # Print the counts of each class in the 'class' column
    print(train_gp['class'].value_counts())


    # Splitting data into train and validation set
    train, val = train_test_split(train_gp, test_size=0.01, stratify=train_gp['class'].tolist())
    
    # Create a list of ImageId for images containing ships
    train_isship_list = train['ImageId'][train['isnan']==0].tolist()

    # Randomly shuffle the list to ensure randomness in the data
    train_isship_list = random.sample(train_isship_list, len(train_isship_list))

    # Create a list of ImageId for images without ships (isnan == 1)
    train_nanship_list = train['ImageId'][train['isnan']==1].tolist()

    # Randomly shuffle the list of images without ships to ensure randomness
    train_nanship_list = random.sample(train_nanship_list, len(train_nanship_list))

    # Set the batch size for the data generator
    BATCH_SIZE = 2

    # Determine the maximum number of images to use (minimum of images with and without ships)
    CAP_NUM = min(len(train_isship_list), len(train_nanship_list))

    # Create an instance of the custom data generator
    datagen = mygenerator(train_df, train_isship_list, train_nanship_list, batch_size=BATCH_SIZE, cap_num=CAP_NUM)

    # Define the U-Net architecture for semantic segmentation
    # Encoder (Contracting Path)
    inputs = Input(shape=(768, 768, 3))
    conv0 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv0 = BatchNormalization()(conv0)
    conv0 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0)
    conv0 = BatchNormalization()(conv0)

    comp0 = AveragePooling2D((6, 6))(conv0)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(comp0)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.4)(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.4)(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.4)(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.4)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    # Decoder (Expansive Path)
    upcv6 = UpSampling2D(size=(2, 2))(conv5)
    upcv6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv6)
    upcv6 = BatchNormalization()(upcv6)
    mrge6 = concatenate([conv4, upcv6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    upcv7 = UpSampling2D(size=(2, 2))(conv6)
    upcv7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv7)
    upcv7 = BatchNormalization()(upcv7)
    mrge7 = concatenate([conv3, upcv7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    upcv8 = UpSampling2D(size=(2, 2))(conv7)
    upcv8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv8)
    upcv8 = BatchNormalization()(upcv8)
    mrge8 = concatenate([conv2, upcv8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    upcv9 = UpSampling2D(size=(2, 2))(conv8)
    upcv9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv9)
    upcv9 = BatchNormalization()(upcv9)
    mrge9 = concatenate([conv1, upcv9], axis=3)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    # Upsample to original input size
    dcmp10 = UpSampling2D((6,6), interpolation='bilinear')(conv9)
    mrge10 = concatenate([dcmp10, conv0], axis=3)
    conv10 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = BatchNormalization()(conv10)

    # Declare the final layer with one filter and sigmoid activation
    conv11 = Conv2D(1, 1, activation='sigmoid')(conv10)

    # Create the model with specified input and output
    model = Model(inputs=inputs, outputs=conv11)

    # Compile the model with the Adam optimizer and binary crossentropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy')


    # Train the model using a generator for 10 epochs with 500 steps per epoch
    history = model.fit_generator(datagen, steps_per_epoch=500, epochs=10)

    # Building a graph to visualize losses on a training set
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    os.makedirs('model', exist_ok=True)
    file_path = os.path.join('model', 'Model_Loss.png')
    plt.savefig(file_path, bbox_inches='tight', dpi=300)

    # If you also have other metrics such as accuracy, you can visualize them as well
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        file_path = os.path.join('model', 'Model_Accuracy.png')
        plt.savefig(file_path, bbox_inches='tight', dpi=300)


    # Save the trained model to a file in the 'model' directory
    file_path = os.path.join('model', 'model.keras')
    model.save(file_path)
    print(f"Model saved successfully at: {file_path}")


def print_versions_libraries():
    print('Library versions')
    print("pandas=={}".format(pd.__version__))
    print("numpy=={}".format(np.__version__))
    print("matplotlib=={}".format(matplotlib.__version__))
    print("keras=={}".format(keras.__version__))
    print("scikit-image=={}".format(skimage.__version__))
    print("scikit-learn=={}\n".format(sklearn.__version__))


def calc_class(area):
    area = area / (768*768)
    if area == 0:
        return 0
    elif area < 0.005:
        return 1
    elif area < 0.015:
        return 2
    elif area < 0.025:
        return 3
    elif area < 0.035:
        return 4
    elif area < 0.045:
        return 5
    else:
        return 6


# These are some helper functions that will help in calculating the ship area and will group them by imageId
# Helper Functions
def rle_to_mask(rle_list, SHAPE):
    tmp_flat = np.zeros(SHAPE[0]*SHAPE[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = rle_list[::2]
        length = rle_list[1::2]
        for i,v in zip(strt,length):
            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 255
        mask = np.reshape(tmp_flat, SHAPE).T
    return mask

def calc_area_for_rle(rle_str):
    rle_list = [int(x) if x.isdigit() else x for x in str(rle_str).split()]
    if len(rle_list) == 1:
        return 0
    else:
        area = np.sum(rle_list[1::2])
        return area
    
# Data generator

def mygenerator(train_df, isship_list, nanship_list, batch_size, cap_num):
    # Select a subset of ImageId for images with and without ships
    train_img_names_nanship = isship_list[:cap_num]
    train_img_names_isship = nanship_list[:cap_num]
    
    k = 0
    while True:
        # Check if the current batch exceeds the limit, then reset the index
        if k + batch_size // 2 >= cap_num:
            k = 0
        
        # Create batches of ImageId for images without ships and with ships
        batch_img_names_nan = train_img_names_nanship[k:k+batch_size//2]
        batch_img_names_is = train_img_names_isship[k:k+batch_size//2]
        
        # Initialize empty lists for images and masks
        batch_img = []
        batch_mask = []
        
        # Process images without ships
        for name in batch_img_names_nan:
            tmp_img = imread(train_dir + name)
            batch_img.append(tmp_img)
            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
            one_mask = np.zeros((768, 768, 1))
            for item in mask_list:
                rle_list = str(item).split()
                tmp_mask = rle_to_mask(rle_list, (768, 768))
                one_mask[:,:,0] += tmp_mask
            batch_mask.append(one_mask)
        
        # Process images with ships
        for name in batch_img_names_is:
            tmp_img = imread(train_dir + name)
            batch_img.append(tmp_img)
            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
            one_mask = np.zeros((768, 768, 1))
            for item in mask_list:
                rle_list = str(item).split()
                tmp_mask = rle_to_mask(rle_list, (768, 768))
                one_mask[:,:,0] += tmp_mask
            batch_mask.append(one_mask)
        
        # Stack images and masks to create batches
        img = np.stack(batch_img, axis=0)
        mask = np.stack(batch_mask, axis=0)
        
        # Normalize pixel values to the range [0, 1]
        img = img / 255.0
        mask = mask / 255.0
        
        # Increment the index for the next batch
        k += batch_size // 2
        
        # Yield the batch of images and masks
        yield img, mask


if __name__ == '__main__':
    main()