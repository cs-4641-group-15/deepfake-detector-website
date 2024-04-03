---
layout: post
title:  "Midterm Report"
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

Our initial step in data preprocessing involved frame extraction so that we could analyze the videos as sequences of individual images. Each video was initially categorized as 'real' or 'fake' based on the given metadata. We selected 24 frames per video, converting them into grayscale to maintain consistency. To standardize the frames for our machine learning model, they were resized and padded to a uniform size of 400x400 pixels using OpenCV, ensuring consistent image formatting across our dataset.

We labeled these frames accordingly and merged them into a comprehensive dataset, which was subsequently divided into training and testing subsets. To refine the input for our model, we calculated the difference between the first two frames of each video and normalized the pixel values to [0, 1]. Additionally, we used TensorFlow's `tf.data.Dataset` tool to structure the data into batches for efficient training.

### Machine Learning Model

We implemented a CNN model that consists of a series of convolutional and max-pooling layers, designed to identify and leverage spatial features in the dataset. Following these, dense layers were added to carry out the classification process. The model uses the Adam optimizer for refining its parameters, with binary cross-entropy as the loss function, chosen for its effectiveness in binary classification tasks.

The training process spanned 20 epochs, during which we closely monitored validation accuracy and loss to ensure the model was generalizing effectively and not overfitting.

## Results and Discussion

The tables below summarize the model’s performance across various metrics during training:

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|-------|-------------------|---------------|---------------------|-----------------|
| 1     | 0.4995            | 2.7291        | 0.4560              | 0.6933          |
| 2     | 0.4734            | 0.6931        | 0.5385              | 0.6916          |
| 3     | 0.5134            | 0.6957        | 0.5220              | 0.6927          |
| 4     | 0.5501            | 0.6890        | 0.4725              | 0.6937          |
| 5     | 0.6356            | 0.6768        | 0.4780              | 0.6955          |
| 6     | 0.6895            | 0.6392        | 0.5000              | 0.6964          |
| 7     | 0.7080            | 0.5789        | 0.4725              | 0.7224          |
| 8     | 0.8132            | 0.4657        | 0.5110              | 0.9043          |
| 9     | 0.8396            | 0.3658        | 0.5165              | 1.1784          |
| 10    | 0.8973            | 0.2324        | 0.5165              | 1.2383          |
| 11    | 0.9369            | 0.1752        | 0.4890              | 1.3145          |
| 12    | 0.9789            | 0.0758        | 0.4780              | 1.5494          |
| 13    | 0.9906            | 0.0426        | 0.4890              | 2.0175          |
| 14    | 0.9903            | 0.0285        | 0.4725              | 1.9173          |
| 15    | 0.9901            | 0.0221        | 0.5330              | 2.0516          |
| 16    | 0.9961            | 0.0240        | 0.5055              | 1.9825          |
| 17    | 0.9944            | 0.0155        | 0.5220              | 2.3383          |
| 18    | 0.9878            | 0.0267        | 0.5165              | 2.4428          |
| 19    | 0.9899            | 0.0269        | 0.5330              | 2.4282          |
| 20    | 0.9996            | 0.0157        | 0.5330              | 2.5089          |

| Metric              | Value                 |
|---------------------|-----------------------|
| Validation Accuracy | 0.5329670310020447    |
| F1 Score            | 0.5893719806763285    |
| AUC-ROC Score       | 0.5257531584062196    |

Notably, we see a constant decrease in the training loss, implying that the model was successfully learning from the dataset. However, the discrepancy between training accuracy and validation accuracy, as well as fluctuating validation loss, indicate possible concerns with the model’s ability to generalize to new data. By the last epoch, the training loss was 0.0157 with a training accuracy of 99.96%, while the validation loss was 2.5089 with a validation accuracy of 53.3%. These results indicate that the model can predict on both training and validation data, but it is heavily overfitted. Despite attempts to mitigate this through the addition of a dropout layer for regularization, further strategies for reducing overfitting are being considered for future improvements, such as artificially adding images to the training set via rotating, flipping, and cropping existing images.

Visual representations of our model’s performance are shown in the graphs below, illustrating the trends in accuracy and loss over the epochs:

![Training and Validation Accuracy](/deepfake-detector-website/images/accuracy.png)
![Training and Validation Loss](/deepfake-detector-website/images/loss.png)

The confusion matrix provides additional insights:

![Confusion Matrix](/deepfake-detector-website/images/confusion_matrix.png)

From this, we can infer the model’s accuracy is approximately 53.2%, indicating that it correctly predicts the class of the video (real or fake) about 53.2% of the time. The precision for fake videos is approximately 42.9%, meaning less than half of the model’s predicted ‘fake’ videos are actually fake, and the recall for fake videos is approximately 49.3%, meaning the model correctly identifies nearly half of all the actual fake videos. 

Moreover, the F1 Score is 0.589, which suggests that the model has a reasonable balance between correctly identifying fake videos and avoiding misclassifications.

![ROC Curve](/deepfake-detector-website/images/roc.png)

The area under the ROC curve (AUC-ROC) is 0.526. For reference, a score of 0.5 indicates random guessing, while a score closer to 1 indicates better performance. Here, the score suggests that the model's ability to discriminate between fake and real videos is slightly better than random guessing, leaving much room for improvement.

In conclusion, our findings underscore the importance of refining both the learning process and the structure of our model to better accommodate the complexities involved in detecting deepfake videos. Our next steps will include exploring more advanced data augmentation techniques, introducing regularization mechanisms, and optimizing model complexity to achieve a more balanced and generalizable performance (preventing overfitting).

## References

[1] T. T. Nguyen et al., "Deep learning for Deepfakes Creation and Detection: A Survey," *Computer Vision and Image Understanding*, vol. 223, no. 103525, p. 103525, Jul. 2022. [Online]. Available: [https://doi.org/10.1016/j.cviu.2022.103525](https://doi.org/10.1016/j.cviu.2022.103525) [Accessed Feb. 22, 2024]

[2] M. Yoon, S.-H. Nam, I.-J. Yu, W. Ahn, M.-J. Kwon, and H.-K. Lee, "Frame-rate up-conversion detection based on convolutional neural network for learning spatiotemporal features," *Forensic Science International*, vol. 340, pp. 111442–111442, Nov. 2022. [Online]. Available: [https://doi.org/10.1016/j.forsciint.2022.111442](https://doi.org/10.1016/j.forsciint.2022.111442) [Accessed Feb. 22, 2024]

[3] A. G. E. Hailtik and W. Afifah, "Criminal Responsibility of Artificial Intelligence Committing Deepfake Crimes in Indonesia," *Asian Journal of Social and Humanities*, vol. 2, no. 4, pp. 776–795, 2023. [Online]. Available: [https://doi.org/10.59888/ajosh.v2i4.222](https://doi.org/10.59888/ajosh.v2i4.222) [Accessed Feb. 22, 2024]

## [Gantt Chart](https://docs.google.com/spreadsheets/d/1EsGv2XncrmJh5mArkpXHedK2YCVcbXae/edit?usp=sharing&ouid=111256331940782469044&rtpof=true&sd=true)

<iframe width="100%" height="500" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQOCWpLpJ2i9XuK1u0PowcwO7Gt4840KjEH3OJ84k_omLtSYR00XxbXqIYXU-SMbA/pubhtml?widget=true&amp;headers=false"></iframe>

## Contributions

| Team Member                      | Responsibilities                                         |
|----------------------------------|----------------------------------------------------------|
| Vibha Thirunellayi Gopalakrishnan| Writing Midterm Report, Evaluation Metrics, Visualizations |
| Junseob Lee                      | Writing Midterm Report, Methods                          |
| Michelle Namgoong                | CNN Model Development, Website Deployment                |
| Yeonsoo Chang                    | Writing Midterm Report, Results, Discussion              |
| Vincent Horvath                  | Data Acquisition, Frame Extraction, Dataset Balancing, CNN Training, Website Deployment |
