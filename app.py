import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Load CIFAR-10 labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255.0
y_test_cat = to_categorical(y_test)

# Load trained model
model = load_model('cifar10.h5')

st.title('CIFAR-10 Model Visualizer')
st.write('Explore the CIFAR-10 dataset and see how your model performs!')

# Sidebar options
option = st.sidebar.selectbox(
    'Choose an action:',
    ['Show random test images', 'Show accuracy/loss curves', 'Upload your own image']
)

if option == 'Show random test images':
    st.header('Random Test Images and Predictions')
    idxs = np.random.choice(len(x_test), 10, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in zip(idxs, axes.flatten()):
        img = x_test[i]
        true_label = class_names[int(y_test[i])]
        pred = model.predict(img[np.newaxis, ...])
        pred_label = class_names[np.argmax(pred)]
        ax.imshow(img)
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis('off')
    st.pyplot(fig)

elif option == 'Show accuracy/loss curves':
    st.header('Training History (Accuracy & Loss)')
    st.write('This feature requires training history to be saved during training.')
    st.info('Retrain your model and save the history to visualize here.')

elif option == 'Upload your own image':
    st.header('Upload an Image for Prediction')
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).resize((32, 32))
        img_arr = np.array(img) / 255.0
        if img_arr.shape != (32, 32, 3):
            st.error('Image must be 32x32 RGB.')
        else:
            pred = model.predict(img_arr[np.newaxis, ...])
            pred_label = class_names[np.argmax(pred)]
            st.image(img, caption=f'Predicted: {pred_label}', use_column_width=True) 