# Mini-Project
## “DETECTION OF TUBERCULOSIS USING RESNET”
### Abstract
Tuberculosis (TB) is a global health concern, and early and accurate detection is essential for
effective disease management. This project proposes a deep learning-based approach for
tuberculosis detection using the ResUNet architecture. By incorporating the advantages of U-
Net and ResNet, the ResUNet model aims to enhance the accuracy and efficiency of
tuberculosis detection from medical images. The ResUNet architecture is designed to
leverage the strengths of U-Net, which is widely used in medical image segmentation, and
ResNet, a popular deep learning architecture known for its ability to mitigate the vanishing
gradient problem. By combining these two architectures, ResUNet aims to capture both local
and global features, allowing for more effective representation of tuberculosis-related patterns
and characteristics in medical images.
#
To facilitate training and evaluation of the ResUNet model, a diverse dataset of chest X-rays
and CT scans, annotated with tuberculosis infection labels, is collected. Preprocessing
techniques, such as resizing, normalization, and enhancement, are applied to ensure
consistent and optimal input for the model.The dataset is then split into training, validation,
and testing subsets. The ResUNet model is trained on the training dataset using optimization
algorithms like stochastic gradient descent or Adam, optimizing the model parameters based
on the comparison of predicted outputs with the ground truth annotations.

During the training process, the model&#39;s performance is assessed using the validation dataset.
This allows for monitoring and fine-tuning of hyperparameters, such as learning rate and
regularization techniques, to achieve optimal model performance and prevent overfitting.
After training, the performance of the trained ResUNet model is evaluated on the testing
dataset. Various evaluation metrics, including accuracy, precision, recall, and F1-score, are
calculated to assess the model&#39;s ability to accurately detect tuberculosis in medical images.

The successful implementation of the ResUNet project for tuberculosis detection holds
promise in improving early diagnosis and treatment outcomes for TB patients. By harnessing
the power of deep learning and combining the strengths of U-Net and ResNet, the ResUNet
model demonstrates its potential as an effective tool for tuberculosis detection from medical
images.
### Introduction
Tuberculosis (TB) remains one of the deadliest infectious diseases globally, with millions of
new cases reported each year. Early detection and prompt treatment are crucial for controlling
the spread of TB and improving patient outcomes. Medical imaging, such as chest X-rays and
CT scans, plays a vital role in the diagnosis and monitoring of TB. However, accurately
interpreting these images can be challenging, requiring specialized expertise and time.To
address this challenge, deep learning techniques have gained significant attention in medical
image analysis, offering the potential to automate and enhance TB detection. Among these
techniques, the ResUNet architecture has emerged as a powerful tool for improving the
accuracy and efficiency of TB detection from medical images.

The ResUNet architecture is a fusion of two well-established deep learning architectures: U-
Net and ResNet. U-Net is renowned for its effectiveness in image segmentation tasks,
enabling the precise delineation of regions of interest. On the other hand, ResNet&#39;s residual
connections address the problem of vanishing gradients, facilitating the flow of information
through the network. By combining the strengths of these architectures, ResUNet aims to
capture both local and global features relevant to TB detection, enabling comprehensive
analysis of medical images.
### Literature Survey
1,V.G. Nair et.al [1] “Detection of Tuberculosis from Chest Radiograph using Deep Learning:
A Systematic Review and Meta-analysis” (2020)
Abstract: Tuberculosis (TB) is a leading cause of mortality and morbidity worldwide. Chest
radiographs (CXRs) are commonly used in TB diagnosis, and machine learning (ML)
algorithms, especially deep learning (DL), have shown promise in CXR-based TB detection.
In this systematic review and meta-analysis, we evaluate the performance of DL-based
algorithms for TB detection from CXR. We found that DL-based algorithms had high
sensitivity and specificity in detecting TB from CXR, with an overall pooled sensitivity of
92.2% and specificity of 90.4%. Our findings suggest that DL-based algorithms have the
potential to improve TB diagnosis, particularly in resource-limited settings.
2,Y. Zhang et al [2],Automatic Tuberculosis Screening System Based on Deep Learning and
Chest X-Ray Images&quot; (2021) aimed to develop an automatic TB screening system using deep
learning and chest X-ray images. The authors recognized the significance of early detection
in effectively treating and controlling tuberculosis (TB), which remains a major global health
problem.In this study, the researchers proposed a system that utilizes a convolutional neural
network (CNN) for the classification of chest X-ray images as either TB-positive or TB-
negative. The deep learning model was trained on a dataset consisting of 662 TB-positive and
662 TB-negative chest X-ray images.
3,H. Azizpour et al. [3] titled &quot;Tuberculosis Detection using Convolutional Neural Networks
with Lung Segmentation&quot; (2019) aimed to propose a deep learning approach for tuberculosis
(TB) detection using convolutional neural networks (CNNs) with lung segmentation. The
authors acknowledged that TB is a significant global public health issue, and accurate and
timely diagnosis is essential for effective treatment and disease control.
In this study, the researchers introduced a deep learning framework that incorporated CNNs
with lung segmentation for TB detection. Lung segmentation is a critical step in isolating the
lung region from chest X-ray images, allowing the model to focus specifically on the areas
relevant to TB diagnosis.

### Objectives
This project aims to delve deeply into the development and implementation of a robust deep
learning model using the ResUNet architecture for tuberculosis detection. The ResUNet
model is a novel combination of the U-Net and ResNet architectures, specifically designed to
enhance the accuracy and efficiency of tuberculosis detection from medical images.To begin,
a comprehensive dataset of chest X-rays and CT scans will be curated, ensuring an adequate
representation of tuberculosis-related patterns and characteristics. This dataset will be
meticulously annotated with labels indicating the presence or absence of tuberculosis
infection. Next, a series of preprocessing steps will be applied to the medical images. This
includes resizing the images to a standardized size, normalizing the pixel values to a common
range, and potentially applying additional enhancements, such as contrast adjustment or noise
reduction. These preprocessing techniques aim to optimize the input images and improve the
model&#39;s ability to detect tuberculosis-related features.

### Problem Statement
The problem statement of this project is to address the challenges in tuberculosis (TB) detection by extending the capabilities of deep learning using the ResUNet architecture. The aim is to develop a robust and accurate system that can effectively analyze medical images, such as chest X-rays or CT scans, for the detection of TB. By leveraging the ResUNet architecture, which combines the strengths of U-Net and ResNet, the project seeks to enhance the accuracy and efficiency of TB diagnosis. The project further aims to overcome the limitations of traditional diagnostic methods, particularly in resource-limited settings where access to specialized healthcare facilities and expertise may be limited. By extending the capabilities of deep learning with ResUNet, the project aspires to provide a reliable and scalable solution for early TB detection, leading to timely interventions, improved patient outcomes, and effective disease control.

### References
1. Suryawanshi, P. B., & Kulkarni, P. S. (2021). Tuberculosis detection using ResNet-50 with lung X-ray
images. International Journal of Scientific Research in Computer Science, Engineering and Information
Technology, 7(2), 466-471.

2. Islam, M. T., & Rahman, M. (2020). Tuberculosis detection from chest X-ray images using deep
learning techniques. In 2020 International Conference on Electrical, Computer and Communication
Engineering (ECCE) (pp. 1-4). IEEE.

3. Kumar, V., & Gautam, A. (2020). Detection of tuberculosis using deep learning models. In 2020
International Conference on Computer Communication and Informatics (ICCCI) (pp. 1-5). IEEE.

4. Ahmed, H. M., Mohamed, M. A., & Zahir, I. (2020). Tuberculosis classification using deep learning
techniques. In 2020 17th International Multi-Conference on Systems, Signals & Devices (SSD) (pp.
79-84). IEEE.

5. Rajpurkar, P., Irvin, J., Ball, R. L., Zhu, K., Yang, B., Mehta, H., ... & Lungren, M. P. (2017). Deep
learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to
practicing radiologists.PLoS medicine, 14(12), e1002686.

6. Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C., Liang, H., Baxter, S. L., ... & Yan, K. (2018).
Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell, 172(5),
1122-1131.

![IMG_20230627_144600](https://github.com/nisartth/Mini-Project/assets/94008426/d9b4687f-c411-40a7-9f9f-d715505d7a7e)



   
