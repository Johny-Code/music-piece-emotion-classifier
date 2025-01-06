# Music Piece Emotion Classifier
This repository contains a Multimodal Deep Learning Model for classifying emotions in music based on audio and lyrics. The model combines audio processing (spectrogram-based architecture) and NLP for lyrics (transformer-based) using late fusion, achieving 60.7% accuracy on the MoodyLyrics4Q dataset.

## Features
* Audio Module: Processes spectrograms with a Sarkar et al.-inspired model.
* Lyrics Module: Uses fine-tuned transformer models for emotion detection.
* Fusion Approaches: Majority voting and concatenation improve multimodal performance.

## Results
* Best Accuracy: 60.7% (majority voting).
* Performs well for "Happy" and "Relaxed" emotions, with challenges for "Sad."
* Robust across diverse musical genres and styles.

### Comparison of Audio Modality Approaches

| **Approach**                | **Accuracy in Literature [%]** | **Accuracy in This Study [%]** |
|------------------------------|---------------------------------|---------------------------------|
| Classical methods (SVM)      | 50                             | 31.38                          |
| Ravdess-based architecture   | 65.96                          | 47.99                          |
| InceptionV3                  | 70–90                          | 53.36                          |
| ResNet                       | 77.36                          | 56.04                          |
| VGG16                        | 63.79                          | 53.69                          |
| Sarkar et al. architecture   | 68–78                          | 59.06                          |
| Inception-ResNet             | 84.91–87.24                    | 56.23                          |

---

### Comparison of Lyric Modality Approaches

| **Approach**                | **Accuracy in Literature [%]** | **Accuracy in This Study [%]** |
|------------------------------|-----------------------------|---------------------------------|
| Feature-based SVM            | 58.0                       | 55.0                           |
| Feature-based ANN            | 58.5                       | 53.9                           |
| Fasttext-based               | -                          | 48.2                           |
| Transformer-based (XLNet)    | 94.78                      | 59.2                           |

---

### Comparison of Joint Classification Approaches

| **Ensemble Approach** | **Precision [%]** | **Recall [%]** | **F1-score [%]** | **Accuracy [%]** |
|------------------------|-------------------|----------------|------------------|------------------|
| Majority voting        | 61.5             | 60.7           | 58.8            | 60.7             |
| Concatenation          | 58.5             | 58.4           | 58.2            | 58.4             |


### Contribution
Copy repository, install necessary requirements using `pip install -r requirements.txt`

### Code formating
In order to run code formatting tool type `autopep8 --global-config .pep8 --in-place --aggressive --aggressive <filepath>`. </br>
Global configuration is placed in the `.pep8` file.