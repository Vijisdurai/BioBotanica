## BioBotanica : Next-Gen Medicinal Plant Detection using Generative AI And RAG.
Medicinal plants play a crucial role in healthcare, research, and conservation. Traditional identification methods are often time-consuming and require expertise.
BioBotanica Next-Gen addresses this with a ResNet model for accurate plant image classification and RAG-based language modeling to provide concise, reliable summaries of medicinal uses.

## About
The BioBotanica Next-Gen project is an AI-driven solution that redefines medicinal plant identification and information accessibility by employing advanced machine learning and language models. Designed to overcome the limitations of traditional plant identification methods, which are time-intensive and require expert knowledge, BioBotanica Next-Gen enables users—whether healthcare professionals, researchers, educators, or plant enthusiasts—to gain valuable botanical insights effortlessly.

## Features
```
- AI-powered plant identification
- Confidence score indicating classification reliability
- Detailed outputs including scientific and common names, medicinal uses
- User-friendly and intuitive interface
- Broad Accessibility
```
## Requirements
# SOFTWARE REQUIREMENTS:
```
Web Application Framework
    -Streamlit: Used for building the interactive web interface that allows users to upload images and view results.
Deep Learning Frameworks:
    -PyTorch: Selected for training and deploying the ResNet model for image classification.
    -Torchvision: Utilized for model architectures and image preprocessing functions.
Natural Language Processing Tools:
    -OpenAI API: Used for generating text and summaries of medicinal properties from the identified plants.
    -Image Processing Library:
    -Pillow: Used for handling and manipulating image inputs from users.
    -HTTP Request Library:
    -Requests: For making HTTP calls to external APIs and information retrieval systems.

Development Environment:
    -IDE/Code Editor: Google Colab, Visual Studio Code  for Python development.
    -Version Control: Git for managing code changes and collaboration.
```
## System Architecture
![WhatsApp Image 2024-11-03 at 22 24 56_27f1cb68](https://github.com/user-attachments/assets/e18521f2-ff9f-498f-988c-8dac8272ff1a)



## Output

#### Output1 - Home Page

![WhatsApp Image 2024-11-03 at 21 29 28_547bcf52](https://github.com/user-attachments/assets/f7d29dbd-b453-49ef-a199-a7a8128b3144)


#### Output2 - Image Classification 

![WhatsApp Image 2024-11-03 at 21 30 41_beb34aea](https://github.com/user-attachments/assets/69a3df06-c6d1-4294-b200-86010ba28a4f)


#### Output3 - Text Generation

![WhatsApp Image 2024-11-03 at 21 31 07_6f1772df](https://github.com/user-attachments/assets/e0a59641-cc77-47b4-bc12-42986f5bc0eb)


Detection Accuracy: 96.7%

## Results and Impact
```
Results-
High Accuracy: ResNet model reliably identifies medicinal plants with confidence scoring.
Concise Information: RAG and GPT-4.0 provide quick summaries of uses, preparations, and safety.
Accessible Interface: Intuitive design.
Fast Processing: Optimized backend ensures quick identification and responses.

Impact-
Supports Healthcare: Aids informed decisions for healthcare and research.
Educational Resource: Makes botanical information accessible for learning.
Improves Accessibility: Democratizes plant knowledge for all users.
Promotes Conservation: Raises awareness for sustainable plant use.
```
## Articles published / References
[1] Vishal Batvia, Drashti Patel, and Dr. Avani R. Vasant, “A Survey on Ayurvedic Medicine Classification using Tensor flow,” International Journal of Computer Trends and Technology, Volume 53, Number 2, November 2017.

[2] Azadnia, Rahim & Al-Amidi, Mohammed & Mohammadi, Hamed & Cifci, Mehmet Akif & Daryab, Avat & Cavallo, Eugenio. (2022). An AI Based Approach for Medicinal Plant Identification Using Deep CNN Based on Global Average Pooling. Agronomy .12.10.3390/ agronomy 12112723. 

[3] Malik OA, Ismail N, Hussein BR, Yahya U. Automated Real-Time Identification of Medicinal Plants Species in Natural Environment Using Deep Learning Models-A Case Study from Borneo Region. Plants (Basel). 2022 Jul 27;11(15):1952. doi: 10.3390/plants11151952. PMID: 35956431; PMCID: PMC9370651.

[4] M.R. Dileep and P.N. Pournami,”Ayurleaf: A deep learning approach for classification of medicinal plants,” IEEE Region 10 Conference (TENCON), 2019.

[5] Ming Liang and Xiaolin Hu, ”Recurrent Convolutional Neural Network for Object Recognition,” IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3367-3375, 2015
