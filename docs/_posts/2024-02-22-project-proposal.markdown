---
layout: post
title:  "Project Proposal"
date:   2024-02-23 9:35:53 -0500
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

### Preprocessing Methods

1. **Frame Extraction:** Use OpenCV’s [VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) tool to extract frames from videos.
2. [**Blob Detection:**](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html) Detect regions in images that differ in properties.
3. **Feature Extraction:** Use [SIFT](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_sift.html) to extract features invariant to affine transformations.

### Machine Learning Models

1. **Convolutional Neural Network (CNN):** Analyze individual frames to extract features (i.e. real or deepfake)
2. **Long Short-Term Memory (LSTM) Network:** Feed in features to capture temporal inconsistencies and create a sequence descriptor representing video content
3. **Fully Convolutional Network (FCN):** Analyze the sequence descriptor to classify videos as real or deepfake

## Potential Results and Discussion

We expect to achieve (and hope to surpass) the 97% accuracy benchmark set by Nguyen et al. [1] by fine-tuning our model. We will prioritize feature engineering to optimize precision and recall for a high F1 Score. Additionally, our focus on accurately classifying videos aims to achieve a strong AUC-ROC performance.

Our goals include refining neural networks to boost detection accuracy and robustness in diverse scenarios, from low-quality to highly convincing deepfakes. We will validate our model's effectiveness on current deepfake technologies and contribute to the discourse on mitigating deepfake-related crimes, informed by Hailtik and Afifah's insights [3].

## References

[1] T. T. Nguyen et al., "Deep learning for Deepfakes Creation and Detection: A Survey," *Computer Vision and Image Understanding*, vol. 223, no. 103525, p. 103525, Jul. 2022. [Online]. Available: [https://doi.org/10.1016/j.cviu.2022.103525](https://doi.org/10.1016/j.cviu.2022.103525) [Accessed Feb. 22, 2024]

[2] M. Yoon, S.-H. Nam, I.-J. Yu, W. Ahn, M.-J. Kwon, and H.-K. Lee, "Frame-rate up-conversion detection based on convolutional neural network for learning spatiotemporal features," *Forensic Science International*, vol. 340, pp. 111442–111442, Nov. 2022. [Online]. Available: [https://doi.org/10.1016/j.forsciint.2022.111442](https://doi.org/10.1016/j.forsciint.2022.111442) [Accessed Feb. 22, 2024]

[3] A. G. E. Hailtik and W. Afifah, "Criminal Responsibility of Artificial Intelligence Committing Deepfake Crimes in Indonesia," *Asian Journal of Social and Humanities*, vol. 2, no. 4, pp. 776–795, 2023. [Online]. Available: [https://doi.org/10.59888/ajosh.v2i4.222](https://doi.org/10.59888/ajosh.v2i4.222) [Accessed Feb. 22, 2024]

## [Gantt Chart](https://docs.google.com/spreadsheets/d/1EsGv2XncrmJh5mArkpXHedK2YCVcbXae/edit?usp=sharing&ouid=111256331940782469044&rtpof=true&sd=true)

<iframe width="100%" height="500" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQOCWpLpJ2i9XuK1u0PowcwO7Gt4840KjEH3OJ84k_omLtSYR00XxbXqIYXU-SMbA/pubhtml?widget=true&amp;headers=false"></iframe>

## Contributions

| Team Member                      | Responsibilities                                     |
|----------------------------------|------------------------------------------------------|
| Vibha Thirunellayi Gopalakrishnan| Problem Definition, Methods, Results, Slide Deck     |
| Junseob Lee                      | Literature Review, Potential Results and Discussion, References |
| Michelle Namgoong                | Final Proposal Edits, GitHub Setup, Slide Deck, Video Presentation |
| Yeonsoo Chang                    | Literature Review, References, Slide Deck            |
| Vincent Horvath                  | Dataset and Description, Preprocessing Methods, Gantt Chart, GitHub Pages Setup and Proposal |
