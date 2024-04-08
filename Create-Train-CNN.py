import tensorflow as tf
import pickle

#### creating the Conventional Neural Network ####

classifier = tf.keras.Sequential()

classifier.add(tf.keras.layers.Convolution2D(128, (3, 3), input_shape=(128, 128, 1), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

classifier.add(tf.keras.layers.Convolution2D(128, (3, 3), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

classifier.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

classifier.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))

classifier.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu'))
classifier.add(tf.keras.layers.GlobalMaxPooling2D())

classifier.add(tf.keras.layers.Dense(units=32, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=4, activation='softmax'))

classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



#### to train the model ####

# Define the directory where the sign images are located
image_directory = 'data'

# Create a data generator with grayscale conversion
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and iterate the training dataset
train_data = datagen.flow_from_directory(
    image_directory,
    target_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

# Load and iterate the validation dataset
validation_data = datagen.flow_from_directory(
    image_directory,
    target_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Fit the model
classifier.fit(
    train_data,
    validation_data=validation_data,
    batch_size=32,
    epochs=70
)



#### give a summary and evaluate the model ####

classifier.summary()

# evaluate model
loss, accuracy = classifier.evaluate(validation_data)



#### Save the model and the labels####


# The labels variable
labels = ['A', 'B', 'C', 'D']

# Saving the labels
with open('model/model-labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

# Saving the model architecture as JSON
model_json = classifier.to_json()
with open("model/model-arch.json", "w") as json_file:
    json_file.write(model_json)


# Saving the model weights as .h5
classifier.save('model/model-weights.h5')

# Print a success message
print("Model architecture and weights were successfully saved.")
