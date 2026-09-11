# 🧠 Emotion Prediction from Text using BiGRU

A Deep Learning and Natural Language Processing (NLP) project that predicts the emotion expressed in a given text sentence using a **Bidirectional GRU (BiGRU)** neural network.

The project compares multiple recurrent neural network architectures and deploys the final model as a **FastAPI web application**.

## 🚀 Live Demo

🔗 **[Emotion Prediction App](https://emotions-prediction-s0bz.onrender.com)**

Enter a sentence and the application predicts one of the six supported emotions.

---

## 🎯 Project Objective

The main objective of this project is to build an NLP-based Deep Learning model that can understand the emotional context of text and classify it into different emotion categories.

### Supported Emotions

- 😢 Sadness
- 😊 Joy
- ❤️ Love
- 😡 Anger
- 😨 Fear
- 😲 Surprise

---

## 📊 Dataset

The project uses the **Emotion Dataset** from Hugging Face:

**Dataset:** `dair-ai/emotion`

The dataset contains text samples belonging to six emotion classes.

The data was explored and analyzed before applying preprocessing and Deep Learning techniques.

---

## 🧹 Text Preprocessing

The text data is converted into numerical sequences before being passed to the neural network.

### Preprocessing Pipeline

```text
Raw Text
   ↓
Tokenization
   ↓
Vocabulary Creation
   ↓
Integer Sequences
   ↓
Padding / Truncation
   ↓
Deep Learning Model
   ↓
Emotion Prediction
