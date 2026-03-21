import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zipfile
import shutil
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_dataset'
app.config['IMAGE_UPLOADS'] = 'static/uploads'
app.secret_key = 'secret123'

# ─── Hyperparameters ───────────────────────────────────────────────
INP_NUM_CLASSES        = 7
INP_EPOCHS             = 30
INP_BATCH_SIZE_TRAIN   = 32
INP_BATCH_SIZE_TEST    = 12
INP_INITIAL_LR         = 0.1
INP_PATIENCE           = 20
# ───────────────────────────────────────────────────────────────────

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_UPLOADS'], exist_ok=True)
os.makedirs('static', exist_ok=True)

MODEL_PATH      = "trained_model.h5"
CONF_MATRIX_PATH = "static/confusion_matrix.png"

model        = None
class_labels = []
training_history = {}


# ══════════════════════════════════════════════════════════════════
#  HOME – upload dataset
# ══════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            flash("No dataset selected!", "error")
            return redirect(request.url)

        shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
        os.makedirs(app.config['UPLOAD_FOLDER'])

        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(dataset_path)

        try:
            with zipfile.ZipFile(dataset_path, "r") as zip_ref:
                zip_ref.extractall(app.config['UPLOAD_FOLDER'])
            os.remove(dataset_path)
        except zipfile.BadZipFile:
            flash("Invalid ZIP file. Please upload a valid dataset.", "error")
            return redirect(request.url)

        flash("Dataset uploaded successfully. Redirecting to training…", "success")
        return redirect(url_for("train_model"))

    return render_template("index.html")


# ══════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════
@app.route("/train", methods=["GET"])
def train_model():
    global model, class_labels, training_history

    dataset_dir = app.config['UPLOAD_FOLDER']

    # ── Data generators ──────────────────────────────────────────
    trdata = ImageDataGenerator(
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1.0 / 255.0
    )

    traindata = trdata.flow_from_directory(
        dataset_dir,
        target_size=(200, 500),
        color_mode="grayscale",
        batch_size=INP_BATCH_SIZE_TRAIN,
        subset="training",
        class_mode="categorical",
        shuffle=True
    )

    testdata = trdata.flow_from_directory(
        dataset_dir,
        target_size=(200, 500),
        color_mode="grayscale",
        batch_size=INP_BATCH_SIZE_TEST,
        subset="validation",
        class_mode="categorical",
        shuffle=False
    )

    # ── Build CNN ─────────────────────────────────────────────────
    model = Sequential([
        Conv2D(filters=2,  kernel_size=(3, 3), padding="same", input_shape=(200, 500, 1)),
        Activation("relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=4,  kernel_size=(3, 3), padding="same"),
        Activation("relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(filters=8,  kernel_size=(3, 3), padding="same"),
        Activation("relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        Flatten(),
        Dense(units=64),
        Activation("relu"),
        Dropout(0.1),
        Dense(units=INP_NUM_CLASSES),
        Activation("softmax")
    ])

    model.summary()

    from keras.optimizers import Adagrad
    opt = Adagrad(learning_rate=INP_INITIAL_LR)
    model.compile(optimizer=opt,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    cnn = model.fit(
        traindata,
        validation_data=testdata,
        epochs=INP_EPOCHS,
        steps_per_epoch=max(1, traindata.samples  // INP_BATCH_SIZE_TRAIN),
        validation_steps=max(1, testdata.samples  // INP_BATCH_SIZE_TEST),
    )

    model.save(MODEL_PATH)

    # ── Evaluate ──────────────────────────────────────────────────
    scores = model.evaluate(testdata,
                            steps=max(1, testdata.samples // INP_BATCH_SIZE_TEST),
                            verbose=1)
    val_accuracy = float(scores[1])
    print(f"Accuracy: {val_accuracy * 100:.2f}%")

    Y_pred   = model.predict(testdata,
                             steps=max(1, testdata.samples // INP_BATCH_SIZE_TEST),
                             verbose=1)
    y_pred   = np.argmax(Y_pred, axis=1)
    class_labels = list(testdata.class_indices.keys())

    from sklearn.metrics import classification_report, confusion_matrix
    cm = confusion_matrix(testdata.classes[:len(y_pred)], y_pred)
    cr = classification_report(testdata.classes[:len(y_pred)], y_pred,
                                target_names=class_labels)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    # ── Confusion matrix plot ──────────────────────────────────────
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH)
    plt.close()

    # ── Accuracy / Loss curves ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(cnn.history['accuracy'],     label='Train Accuracy')
    axes[0].plot(cnn.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(cnn.history['loss'],     label='Train Loss')
    axes[1].plot(cnn.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('static/training_curves.png')
    plt.close()

    training_history = cnn.history

    cm_img        = url_for('static', filename='confusion_matrix.png')
    training_img  = url_for('static', filename='training_curves.png')

    return render_template(
        "train.html",
        accuracy=val_accuracy,
        cm_img=cm_img,
        training_img=training_img,
        cr=cr,
        class_labels=class_labels,
    )


# ══════════════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["GET", "POST"])
def predict_image():
    global model, class_labels

    if model is None:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            flash("Train a model first before making predictions.", "error")
            return redirect(url_for("upload_dataset"))

    if request.method == "POST":
        if "imagefile" not in request.files:
            flash("No file part", "error")
            return redirect(request.url)

        file = request.files["imagefile"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)

        if file:
            img_path = os.path.join(app.config['IMAGE_UPLOADS'], file.filename)
            file.save(img_path)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                flash("Could not read image. Please upload a valid image file.", "error")
                return redirect(request.url)

            img = cv2.resize(img, (500, 200))          # (width, height)
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=[0, -1])    # (1, 200, 500, 1)

            predictions    = model.predict(img)
            pred_idx       = int(np.argmax(predictions))
            confidence     = float(np.max(predictions)) * 100
            predicted_class = class_labels[pred_idx] if class_labels else f"Class {pred_idx}"

            all_probs = {
                class_labels[i] if class_labels else f"Class {i}": float(predictions[0][i]) * 100
                for i in range(len(predictions[0]))
            }

            return render_template(
                "predict.html",
                image_file=file.filename,
                prediction=predicted_class,
                confidence=f"{confidence:.2f}",
                all_probs=all_probs,
            )

    return render_template("predict.html", image_file=None, prediction=None,
                           confidence=None, all_probs=None)


if __name__ == "__main__":
    app.run(debug=True)
