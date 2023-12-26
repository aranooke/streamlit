import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from transformers import pipeline, DistilBertTokenizer, TFDistilBertForSequenceClassification
sentiment_analyzer = pipeline('sentiment-analysis')
# Load the MobileNetV2 model from TensorFlow Hub for image classification
image_model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load pre-trained DistilBERT model for sentiment analysis
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
text_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
text_classifier = pipeline('sentiment-analysis', model=text_model, tokenizer=tokenizer)

# Load pre-trained DeepLabV3 model for image segmentation
deeplab_model = tf.keras.applications.DenseNet201(weights='imagenet')

# Function to preprocess and predict with the image classification model
def predict_image(image, model):
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (224, 224))
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    prediction = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(prediction)[0]

    return decoded_predictions

# Function to predict sentiment with the text classification model
def predict_text_sentiment(text):
    result = text_classifier(text)[0]
    return result

# Function to preprocess and predict with the image segmentation model
def predict_image_segmentation(image, model):
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (224, 224))
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.densenet.preprocess_input(img_array)

    prediction = model.predict(img_array)
    # You need to customize how to interpret the segmentation result based on the model's output
    # This is just a placeholder, and you might need post-processing steps
    segmented_image = prediction[0][:, :, 0]

    return segmented_image

# Streamlit app
def main():
    st.title("Mixed Image and Text Classification")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Input text
    text_input = st.text_input("Enter some text for sentiment analysis")

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Predict image classification
        image_prediction = predict_image(image, image_model)
        st.write("Image Classification Prediction:")
        for i, (imagenet_id, label, score) in enumerate(image_prediction):
            st.write(f"{i + 1}: {label} ({score:.2f})")

        # Predict image segmentation (using placeholder result)
        image_segmentation = predict_image_segmentation(image, deeplab_model)
        st.write("Image Segmentation Prediction:")
        st.image(image_segmentation, caption="Segmented Image", use_column_width=True, channels="GRAY")

    if text_input:
        # Predict sentiment
        text_sentiment = sentiment_analyzer(text_input)[0]
        st.write("Text Sentiment Analysis:")
        st.write(f"Label: {text_sentiment['label']}")
        st.write(f"Score: {text_sentiment['score']:.2f}")

if __name__ == "__main__":
    main()
