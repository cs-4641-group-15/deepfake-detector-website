---
layout: post
title:  "Final Report"
date:   2024-04-23 07:28:19 -0500
categories: update
---

## Introduction/Background

### Literature Review

The advent of deep learning has led to the emergence of 'deepfakes,' synthetic media where a person's likeness can be swapped or manipulated with high realism, posing significant risks to privacy, security, and information authenticity. As deepfakes become more sophisticated, detecting them becomes critical.

In the literature, Nguyen et al. [1] survey deep learning techniques for generating and detecting deepfakes, delineating the evolution of detection methodologies and setting benchmarks for detection accuracy. Further, Yoon et al. [2] demonstrate the effectiveness of convolutional neural networks (CNNs) in identifying frame-rate inconsistencies -- a technique that could be vital for deepfake detection.

The societal implications of deepfakes, particularly legal and ethical considerations, are discussed by Hailtik and Afifah [3], emphasizing the need for a legal framework to address AI-generated deepfake crimes. Our project builds upon these insights, aiming to develop a robust detection system capable of classifying deepfaked content with high accuracy. By doing so, we hope to contribute to the safer use of artificial intelligence (AI) in digital media creation.

## Dataset Description

Our [dataset](https://www.kaggle.com/c/deepfake-detection-challenge/data) comes from the Deepfake Detection Challenge jointly hosted on Kaggle by AWS, Microsoft, and Meta. It features more than 470 GB of data, including over 23K real and 100K deepfaked .mp4 videos, distributed across 50 .zip files each containing a .JSON file that indicates which are real or fake, as well as which are testing or training data.

## Problem Definition

Deepfake technology has led to loss of trust in media content, distress to targeted individuals, and greater spread of misinformation. With increasing accessibility to generative tools, such as [Deepfacelab](https://github.com/iperov/DeepFaceLab), there is an urgent need to develop effective deepfake detection methods to combat these detrimental effects.

## Methods

### Data Preprocessing

#### Frame Extraction

Our initial step in data preprocessing involved *frame extraction* so that we could convert each video into analyzable image sequences. These sequences were labeled as 'real' or 'fake' based on the given metadata. We selected 24 frames from each video, converting them into grayscale to maintain consistency. 

To standardize the frames for our machine learning model, they were resized and padded to a uniform size of 400x400 pixels using OpenCV, ensuring consistent image formatting across our dataset. We labeled these frames accordingly and merged them into a comprehensive dataset, which was subsequently divided into training and testing subsets. To refine the input, we calculated the difference between the first two frames of each video and normalized the pixel values to [0, 1]. Additionally, we used TensorFlow's `tf.data.Dataset` tool to structure the data into manageable batches for efficient training.

#### Dimensionality Reduction

Our next step in data preprocessing involved using *principal component analysis (PCA)* as a dimensionality reduction method. Given the size and complexity of our dataset, we aimed to reduce computation complexity while retaining the most informative aspects of the data. PCA helped reduce the dataset, converting the frames into feature vectors that capture the most variance within the data.

### Machine Learning Models

We designed a *convolutional neural network (CNN)* with a layered architecture, including convolutional layers for feature extraction and dense layers for classification. The CNN is well-suited for capturing spatial features in image data, making it ideal for our project where the spatial arrangement of pixels is crucial to deepfake detection.

To capture temporal dynamics between frames, we implemented a *long short-term memory (LSTM)* network. LSTMs are a type of recurrent neural network that can model sequential data so that outputs are not only a function of the current input but also of the prior input. Such functionality is particularly useful for our project, as temporal inconsistencies could hint at video manipulation.

Complementary to the CNN and LSTM, we built a *fully connected network (FCN)*. FCNs can be effective at classification tasks due to their simple architecture, which allows for a focus on combining features in a nonlinear fashion. Since our project aims to classify videos as real or fake, we thought that these models would yield the most success.

All models were trained using the Adam optimizer and binary cross-entropy loss, chosen for their effectiveness in binary classification tasks. Additionally, the use of early stopping ensured that our models stopped training once the validation loss failed to improve, preventing them from learning noise in the training data.

### Training and Evaluation

Our training process involved supervised learning, with each model learning directly from labeled ‘real’ or ‘fake’ data. To guide any necessary adjustments, each model was evaluated based on accuracy, F1 score, and AUC-ROC (all within [0, 1]), where high accuracy denotes correct predictions, high F1 score denotes a good balance between correctly identifying deepfake videos (precision) and avoiding misclassifications (recall), and high AUC-ROC denotes good classification capability (0.5 indicates random guessing). By analyzing these metrics, we aimed to achieve optimal performance for each model.

## Results and Discussion

### Convolutional Neural Network (CNN)

The CNN’s performance is summarized in the table below:

| Metric              | Value                 |
|---------------------|-----------------------|
| Validation Accuracy | 0.5329670310020447    |
| F1 Score            | 0.5893719806763285    |
| AUC-ROC Score       | 0.5257531584062196    |

The training and validation accuracy and loss graphs illustrate trends in the model’s performance. A consistent decrease in training loss indicates effective learning from the dataset. However, a significant divergence between training and validation accuracy points to heavy overfitting.

![Training and Validation Accuracy](/deepfake-detector-website/images/cnn_accuracy.png)
![Training and Validation Loss](/deepfake-detector-website/images/cnn_loss.png)

Based on the confusion matrix, we can infer that the model’s accuracy is 53.3%. The precision for fake videos at around 42.9% suggests that less than half of the predicted ‘fake’ videos are truly fake, while the recall of about 49.3% indicates that the model identifies just under half of all actual fake videos. Moreover, an F1 score of 0.589 implies a moderate balance between precision and recall.

![Confusion Matrix](/deepfake-detector-website/images/cnn_cmatrix.png)

Additionally, the ROC curve has an AUC of 0.526, only slightly outperforming random guessing.

![ROC Curve](/deepfake-detector-website/images/cnn_roc.png)

These results highlight the CNN’s limited ability in classifying deepfake videos.

### Long Short-Term Memory (LSTM) Network

The LSTM's performance is summarized in the table below:

| Metric              | Value                 |
|---------------------|-----------------------|
| Validation Accuracy | 0.5549450516700745    |
| F1 Score            | 0.5714285714285714    |
| AUC-ROC Score       | 0.569120505344995     |

The LSTM’s accuracy and loss over epochs, as shown in the following graphs, exhibit fluctuations without substantial improvement in validation accuracy. These erratic patterns suggest instability in the model’s learning and potential challenges in capturing the temporal dynamics of videos. Additionally, there is a large gap between training and validation  accuracy and loss.

![Training and Validation Accuracy](/deepfake-detector-website/images/lstm_accuracy.png)
![Training and Validation Loss](/deepfake-detector-website/images/lstm_loss.png)

The confusion matrix indicates an accuracy of 55.5%. The precision for fake videos is about 56.9%, with the model’s predicted ‘fake’ videos being actually fake more than half of the time, and the recall for fake videos is around 51.6%, where the model identifies slightly more than half of the actual fake videos. Moreover, an F1 score of 0.571 suggests that the model has a moderate balance between precision and recall.

![Confusion Matrix](/deepfake-detector-website/images/lstm_cmatrix.png)

Furthermore, the ROC curve’s AUC of 0.569 denotes a modest improvement over random guessing but still not highly effective.

![ROC Curve](/deepfake-detector-website/images/lstm_roc.png)

These results reflect the LSTM’s moderate capability in classifying deepfake videos.

### Fully Connected Network (FCN)

The FCN’s performance is summarized in the table below:

| Metric              | Value                 |
|---------------------|-----------------------|
| Validation Accuracy | 0.5549450516700745    |
| F1 Score            | 0.5970149253731343    |
| AUC-ROC Score       | 0.5334062196307094    |

The FCN demonstrates high training accuracy yet low, fluctuating validation accuracy in the accuracy and loss graphs, indicating potential overfitting. Nevertheless, convergence of training and validation loss is more pronounced, suggesting relatively mild overfitting than in the CNN. These patterns reflect successful learning.

![Training and Validation Accuracy](/deepfake-detector-website/images/fcn_accuracy.png)
![Training and Validation Loss](/deepfake-detector-website/images/fcn_loss.png)

From the confusion matrix, we interpret the model’s accuracy as 55.5%,  meaning it correctly predicts the class of the video (real or fake) about 55.5% of the time. With a precision for fake videos of around 48.8%, the model labels fake videos correctly about half the time, and a recall of 53.2% signifies it identifies just over half of all actual fake videos. Moreover, an F1 score of 0.597 implies a relatively strong balance between precision and recall. 

![Confusion Matrix](/deepfake-detector-website/images/fcn_cmatrix.png)

Additionally, the ROC curve presents an AUC of 0.533, demonstrating a slight advantage over random guessing.

![ROC Curve](/deepfake-detector-website/images/fcn_roc.png)

These findings emphasize the FCN’s limited performance in classifying deepfake videos.

### Comparison

Each model exhibited unique strengths and weaknesses. The FCN, with the highest F1 score, provides a more balanced precision and recall approach. Yet, all models struggle with generalization, affecting accuracy. The LSTM, designed to capture temporal dynamics in videos, marginally surpasses the CNN in the AUC-ROC score, pointing to limited success in temporal modeling. Such limitations could stem from using PCA, which may have ignored essential temporal features necessary for deepfake detection. The CNN, proficient in static image feature extraction, exhibits the greatest disparity between training and validation accuracies, a clear sign of overfitting. Both the LSTM and FCN models, with more evenly distributed error rates in their confusion matrices, seem better equipped to handle class imbalance than the CNN. Still, the simpler architecture of the LSTM and FCN may lack the complexity needed to effectively distinguish between real and fake videos.

## Conclusion

Accurately detecting deepfake videos proved to be a difficult task. Our observations underscore the need for refined learning processes and model structures to effectively handle the complexities of deepfake detection. While each model demonstrates inherent strengths -- CNN's feature extraction, LSTM's temporal processing, and FCN's balance -- their performances suggest a need for further optimization. Future strategies could involve enhancing feature extraction to better capture temporal relationships, managing class imbalance with data augmentation, and implementing more advanced regularization techniques to prevent overfitting. With more time and resources, we could explore building an ensemble model that combines the strengths of each individual approach to improve overall classification performance.

## References

[1] T. T. Nguyen et al., "Deep learning for Deepfakes Creation and Detection: A Survey," *Computer Vision and Image Understanding*, vol. 223, no. 103525, p. 103525, Jul. 2022. [Online]. Available: [https://doi.org/10.1016/j.cviu.2022.103525](https://doi.org/10.1016/j.cviu.2022.103525) [Accessed Feb. 22, 2024]

[2] M. Yoon, S.-H. Nam, I.-J. Yu, W. Ahn, M.-J. Kwon, and H.-K. Lee, "Frame-rate up-conversion detection based on convolutional neural network for learning spatiotemporal features," *Forensic Science International*, vol. 340, pp. 111442–111442, Nov. 2022. [Online]. Available: [https://doi.org/10.1016/j.forsciint.2022.111442](https://doi.org/10.1016/j.forsciint.2022.111442) [Accessed Feb. 22, 2024]

[3] A. G. E. Hailtik and W. Afifah, "Criminal Responsibility of Artificial Intelligence Committing Deepfake Crimes in Indonesia," *Asian Journal of Social and Humanities*, vol. 2, no. 4, pp. 776–795, 2023. [Online]. Available: [https://doi.org/10.59888/ajosh.v2i4.222](https://doi.org/10.59888/ajosh.v2i4.222) [Accessed Feb. 22, 2024]

## [Gantt Chart](https://docs.google.com/spreadsheets/d/1EsGv2XncrmJh5mArkpXHedK2YCVcbXae/edit?usp=sharing&ouid=111256331940782469044&rtpof=true&sd=true)

<iframe width="100%" height="500" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQOCWpLpJ2i9XuK1u0PowcwO7Gt4840KjEH3OJ84k_omLtSYR00XxbXqIYXU-SMbA/pubhtml?widget=true&amp;headers=false"></iframe>

## Contributions

| Team Member                      | Responsibilities                                         |
|----------------------------------|----------------------------------------------------------|
| Vibha Thirunellayi Gopalakrishnan| FCN, Evaluation and Metrics, Video Script                |
| Junseob Lee                      | Final Report, Video                                      |
| Michelle Namgoong                | Attempt at CNN-LSTM-FCN Model, LSTM, Final Report, Slide Deck |
| Yeonsoo Chang                    | Final Report, Slide Deck                                 |
| Vincent Horvath                  | PCA, LSTM, FCN, Evaluation and Metrics |
