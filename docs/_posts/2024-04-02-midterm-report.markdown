---
layout: post
title:  "Midterm Report (DRAFT)"
date:   2024-04-01 00:35:53 -0500
categories: update
---


## Introduction/Background

### Literature Review

The advent of deep learning has led to the emergence of 'deepfakes,' synthetic media where a person's likeness can be swapped or manipulated with high realism, posing significant risks to privacy, security, and information authenticity. As deepfakes become more sophisticated, detecting them becomes critical.

In the literature, Nguyen et al. [1] survey deep learning techniques for generating and detecting deepfakes, delineating the evolution of detection methodologies and setting benchmarks for detection accuracy. Complementary to this, Yoon et al. [2] demonstrate the effectiveness of convolutional neural networks (CNNs) in identifying frame-rate inconsistencies—a technique that could be vital for deepfake detection.

The societal implications of deepfakes, particularly legal and ethical considerations, are discussed by Hailtik and Afifah [3], emphasizing the need for a legal framework to address AI-generated deepfake crimes. Our project builds upon these insights, aiming to develop a robust detection system capable of classifying deepfaked content with high accuracy. Through this, we aspire to contribute to the safer use of artificial intelligence (AI) in digital media creation.

## Dataset Description

Our [dataset](https://www.kaggle.com/c/deepfake-detection-challenge/data) comes from the Deepfake Detection Challenge jointly hosted on Kaggle by AWS, Microsoft, and Meta. It features more than 470 GB of data, including over 23K real and 100K deepfaked .mp4 videos, distributed across 50 .zip files each containing a .JSON file that indicates which are real or fake, as well as which are testing or training data.

## Problem Definition

Deepfake technology has led to loss of trust in media content, distress to targeted individuals, and greater spread of misinformation. With increasing accessibility to generative tools, such as [Deepfacelab](https://github.com/iperov/DeepFaceLab), there is an urgent need to develop effective deepfake detection methods to combat these detrimental effects.

## Methods

### Data Preprocessing

Videos in our dataset are initially classified as 'real' or 'fake' based on metadata. We capture 24 frames from each video and convert them to grayscale before scaling and padding to produce uniform 400x400 pixel images. This method is carried out using OpenCV, which ensures that the dataset has uniformly formatted images for CNN input.

Extracted frames are labeled accordingly and merged into a comprehensive dataset, which is then split into training and testing subsets. We average the frames of each video to simplify model input and normalize pixel values to [0, 1] for improved model performance. The data is structured into batches using TensorFlow's `tf.data.Dataset` for efficient training.

### Machine Learning Model

Our model integrates a sequence of convolutional and max-pooling layers, thoroughly designed to detect and harness spatial features within the dataset. Following these, dense layers are employed to facilitate the classification process. Optimization is achieved through the use of the 'adam' optimizer, while 'binary cross-entropy' serves as the loss function, closely selected for its efficacy in binary classification scenarios.

The model is trained over 10 epochs with real-time data augmentation and performance monitored via validation accuracy and loss. This iterative process ensures the model is generalizing well and not overfitting.

## Results and Discussion

| Epoch | Loss   | Accuracy | Val Loss | Val Accuracy |
|-------|--------|----------|----------|--------------|
| 1     | 0.5535 | 0.8001   | 0.5485   | 0.7996       |
| 2     | 0.4837 | 0.8151   | 0.5074   | 0.7996       |
| 3     | 0.4761 | 0.8151   | 0.4953   | 0.7996       |
| 4     | 0.4683 | 0.8151   | 0.5018   | 0.7996       |
| 5     | 0.465  | 0.8151   | 0.5236   | 0.7996       |
| 6     | 0.4652 | 0.814    | 0.5473   | 0.7996       |
| 7     | 0.4592 | 0.8146   | 0.5057   | 0.7996       |
| 8     | 0.4544 | 0.8151   | 0.5106   | 0.7996       |
| 9     | 0.4597 | 0.8166   | 0.5519   | 0.7996       |
| 10    | 0.4443 | 0.8146   | 0.5495   | 0.7996       |

In our evaluation of the CNN model's performance, we may reach several conclusions. Initially, we saw a constant decrease in training loss, implying that the model was efficiently learning from the training data. This indicates that the learning process using the training data is efficient. However, the variety in validation loss and the consistency of training accuracy vs variation in validation accuracy indicate possible concerns with the model's capacity to generalize to new data. At the final epoch, the training loss was recorded at 0.4443, with a training accuracy of 81.46%, while the validation loss stood at 0.5495, with a validation accuracy of 79.96%. The results show that the model can predict on both training and validation data, but the difference in training and validation accuracy increases the risk of overfitting.

To prevent overfitting and improve our CNN model's generalization, we may use tactics such as data augmentation, regularization, and model complexity modifications. These techniques try to improve performance in a variety of circumstances and our findings emphasize the need of improving the model's learning process and structure.


## References

[1] T. T. Nguyen et al., "Deep learning for Deepfakes Creation and Detection: A Survey," *Computer Vision and Image Understanding*, vol. 223, no. 103525, p. 103525, Jul. 2022. [Online]. Available: [https://doi.org/10.1016/j.cviu.2022.103525](https://doi.org/10.1016/j.cviu.2022.103525) [Accessed Feb. 22, 2024]

[2] M. Yoon, S.-H. Nam, I.-J. Yu, W. Ahn, M.-J. Kwon, and H.-K. Lee, "Frame-rate up-conversion detection based on convolutional neural network for learning spatiotemporal features," *Forensic Science International*, vol. 340, pp. 111442–111442, Nov. 2022. [Online]. Available: [https://doi.org/10.1016/j.forsciint.2022.111442](https://doi.org/10.1016/j.forsciint.2022.111442) [Accessed Feb. 22, 2024]

[3] A. G. E. Hailtik and W. Afifah, "Criminal Responsibility of Artificial Intelligence Committing Deepfake Crimes in Indonesia," *Asian Journal of Social and Humanities*, vol. 2, no. 4, pp. 776–795, 2023. [Online]. Available: [https://doi.org/10.59888/ajosh.v2i4.222](https://doi.org/10.59888/ajosh.v2i4.222) [Accessed Feb. 22, 2024]

## [Gantt Chart](https://docs.google.com/spreadsheets/d/1EsGv2XncrmJh5mArkpXHedK2YCVcbXae/edit?usp=sharing&ouid=111256331940782469044&rtpof=true&sd=true)

<iframe width="100%" height="500" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQOCWpLpJ2i9XuK1u0PowcwO7Gt4840KjEH3OJ84k_omLtSYR00XxbXqIYXU-SMbA/pubhtml?widget=true&amp;headers=false"></iframe>

## Contributions TODO: Update

| Team Member                      | Responsibilities                                     |
|----------------------------------|------------------------------------------------------|
| Vibha Thirunellayi Gopalakrishnan| Midterm Written Portion, Accuracy Evaluation    |
| Junseob Lee                      | Collaborative work on CNN, Lead Midterm Checkpoint Draft |
| Michelle Namgoong                | CNN Model Development, Initial Model Assessment |
| Yeonsoo Chang                    | Analyzing Loss and Accuracy, Visualization, Midterm Checkpoint            |
| Vincent Horvath                  | Data Acquisition, Frame Extraction, Dataset Balancing, CNN Training, Website Deployment |
