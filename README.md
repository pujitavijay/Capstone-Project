# Replay Attack Detection in IoT Using Deep Learning

## Project Overview
This project addresses the significant cybersecurity risk of replay attacks in IoT devices. Despite ongoing improvements in IoT device security, effectively stopping replay attacks remains a challenge. The project involves setting up IoT communication networks using XCTU software, applying deep learning models for classification, and analyzing the effectiveness of these models in detecting replay attacks.

## Tools and Technologies
- **XCTU Software**: Used for setting up IoT communication networks, such as Zigbee networks.
- **Deep Learning Models**: Various models, including Long Short-Term Memory (LSTM) networks, are utilized for intrusion detection.
- **Python**: Programming language used for implementing deep learning models.
- **Libraries**: 
  - **TensorFlow/Keras**: For building and training deep learning models
  - **scikit-learn**: For preprocessing and evaluation
  - **NumPy** and **pandas**: For data manipulation

## Dataset
The dataset includes records of regular transmissions and replay attacks collected during IoT communications. It is used to train and evaluate deep learning models for intrusion detection.

## Deep Learning Models
- **LSTM & Bi-LSTM Model**: Applied to recognize intrusions in the context of replay attacks.
- **Loss Function, Optimizer, and Metrics**: Defined to process and examine the dataset effectively.
