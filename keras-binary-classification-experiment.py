
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print(tf.__version__)
print(keras.__version__)

training_data = pd.read_csv('/Users/emelyebarlow/Desktop/github-projects/Wine-AI-Keras-Eval-matrics/wine-3-1.csv')

training_y = training_data.pop('quality')
training_y.replace('good', 1, inplace=True)
training_y.replace('bad', 0, inplace=True)
training_x = training_data

arr_convert_x = training_x.to_numpy()
arr_convert_y = training_y.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(arr_convert_x, arr_convert_y, test_size=0.4, random_state=2, shuffle=True)

from tensorflow.keras.callbacks import EarlyStopping

model = keras.Sequential([
    layers.InputLayer(shape=(11,)),
    layers.Dense(128, activation= 'relu'),
    layers.Dense(64, activation='relu'), #2 hidden layers seems to make model more accurate
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model_training_history = model.fit(X_train, Y_train, epochs=60, validation_data=(X_test, Y_test))

# early_stop = EarlyStopping(
#     monitor='val_loss', 
#     patience=10,  # wait 10 epochs before stopping
#     restore_best_weights=True
# )

# history = model.fit(
#     training_x, training_y,
#     validation_data=(validation_X, validation_y),
#     epochs=100,
#     callbacks=[early_stop]
# )


plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

ax1.plot(model_training_history.history['accuracy'], label='Training Accuracy')
ax1.plot(model_training_history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epochs')
ax1.legend()

ax2.plot(model_training_history.history['loss'], label='Training Loss')
ax2.plot(model_training_history.history['val_loss'], label='Validation Loss', linestyle='--')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epochs')
ax2.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
print("Training plots saved as training_metrics.png")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

test_loss, test_acc = model.evaluate(X_test, Y_test)
print(test_loss, test_acc)

y_pred = model.predict(X_test)

print(confusion_matrix(Y_test, np.round(y_pred)))


tn, fp, fn, tp = confusion_matrix(Y_test, np.round(y_pred)).ravel()
print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"TN: {tn}")
print(f"FN: {fn}")

from sklearn.metrics import ConfusionMatrixDisplay 

ConfusionMatrixDisplay.from_predictions(Y_test, np.round(y_pred))
y_pred = np.round(model.predict(X_test))
cm = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")

