# (CSE 151A Project) From Leaves to Labels: Classifying Houseplants with Machine Learning

Group members:
- Catherine Du (c5du@ucsd.edu)
- Brenton Dunn (bmdunn@ucsd.edu)
- Matthew Tan (mztan@ucsd.edu)
- Trisha Tong (trtong@ucsd.edu)
- Sophia Yu (soy001@ucsd.edu)

## Environment Setup

Create virtual environment:
```
python -m venv .venv
```

Activate virtual environment:
```
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Download data:

(1)
```python
python download_data.py
```
The path to where the data is will be printed to the console. 


(2) Download from [Kaggle](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species).

(3) Download using `opendatasets`

```
# in a notebook
!pip install opendatasets

import opendatasets as od
import pandas as pd

od.download(
    "https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species")
```

The parent folder for the data should be called `house_plant_species`. Move the data (either with `mv` or with GUI) so that the file structure looks like the following:
- fa24-cse151a-project (current working directory)
  - house_plant_species
  - download_data.py
  - data_exploration.ipynb
  - requirements.txt
 
<details><summary><h1>Previous Milestone Submissions (click dropdown)</h1></summary>
	
## Data Exploration
This dataset was sourced from Kaggle and contains 14,790 images of various houseplant species across 47 classes. 

**Key Features**
- Number of images: 14,790
- Number of classes: 47
- Source: Bing Images
- Curation: Manually curated by a non-professional biologist
- Organization: Images are organized into folders named after each plant species

**Image Characteristics**
- Variability in quality, resolution, and size
- Taken in both indoor and outdoor settings
- Includes both close-ups of features and depictions of whole plants

## Preprocessing Steps
We are working with an image dataset containing 14,790 images of varying quality, resolution, and size. Given these parameters, as well as the details gathered from our data exploration step, we plan to perform the following preprocessing steps to the dataset before training our model:

1. Check for and remove any corrupted images
   - We do not want corrupted input data to negatively impact our model during training
3. Resize images to 224x224 pixels
   - Having a consistent input size is crucial for our model to operate efficiently
4. Create label encoding and one-hot encoding for each of the 47 plant classes
   - One-hot encoding is more commonly used for non-ordinal categories (such as our different plant classes), it is also more computationally expensive and may lead to the curse of dimensionality
   - Label-encoding is more general and could be more efficient, but may lead to unintentionally creating a false ordering of classes
   - We plan to experiment with both types of encoding to see which one makes the most sense for our use case

<hr>

**All of our work for Milestone 3 is in `MS3.ipynb`.**

## Evaluate your model and compare training vs. test error

We decided to use accuracy to evaluate our model. We saw training accuracy around 29% and testing accuracy around 19%. We noticed that training is higher than test (more in the conclusion).

## Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?

We are possibly underfitting with a too simple model since our:
- Training data accuracy is around 0.2900
- Validation accuracy is around 0.1866
- Test accuracy is around 0.1945
- Baseline loss would be around 3.85, and since our loss reach 2.644 we are improving but ideally loss would be closer to 1
- Even though the training dataset is higher in accuracy than the validation and test, holistically speaking, we would need to run the epochs more to capture the data better.
  ![image](https://github.com/user-attachments/assets/477c517e-4d5f-4e15-86f4-43bb1601cdd2)



We could try:
- Deeper CNN model since the additional layers may be able to catch smaller details like certain patterns on the leaves
- CNN with attention so that it could focus on more important parts, also like the patterns on leaves.
- Pretrained models with transfer learning since it can reduce the training time since our model takes a while to train, and improve accuracy at the same time.

## New work and updates

Major items:
- Added CNN model
- Fixed pre-processing suggestions in MS2 feedback (kept color instead of grayscale for images)
- Created test, train, and validation sets

## What is the conclusion of your 1st model? What can be done to possibly improve it?

The conclusion of our 1st model is:
- It has an accuracy higher than random since random would only be only be 2.12%. Even though the test accuracy is higher than the train accuracy and one may think that we are overfitting, since in the wider picture both accuracies are relatively low, we believe that we are underfitting the model. This makes sense since due to compute limitations, we had limited hyperparameter tuning.

We could improve it by:
- Increasing the number of epochs
- Add data augmentation by adding transformations (torchvision.transforms, flips, scaling)
- Change learning rate
- Regularization (dropout, weight decay)
- Change network architecture parameters


# Ground Truth and Example Predictions for Train, Validation, and Test
![image](https://github.com/user-attachments/assets/551d307a-5f93-4fb8-8472-5046451755e3)




<hr>

# MS4

**1: Train your second model. Make sure you use a different model than in MS3, and you must fine-tune your model to get an accurate comparison.**

Our MS3 model was a CNN and our MS4 model is SVM. In order to ensure that we get an accurate comparison, we made sure that our train/test/split ratios stayed the same and that we used the same random seed to make the split.

**2: Evaluate your model and compare training vs. test error**

After a lot of hyperparameter tuning, the model we decided to go with had the following hyperparameters:
- HOG feature extraction with parameters:
  - orientation=12
  - pixels_per_cell=(16,16)
  - cells_per_block=(2,2)
  - block_norm=’L2-Hys’
  - Scaled via StandardScaler
- SVM with an rbf kernel 
  - C=1
  - 
This provided the following accuracy:

Train: 87%

Test: 27%

```
SVM Train Accuracy: 0.8716655349694442
Classification report:           	precision	recall  f1-score   support

       	0   	0.91  	0.86  	0.88   	247
       	1   	0.99  	0.87  	0.93   	203
       	2   	0.94  	0.90  	0.92   	228
       	3   	1.00  	0.83  	0.91   	131
       	4   	0.99  	0.78  	0.87   	117
       	5   	0.84  	0.79  	0.82   	186
       	6   	0.88  	0.96  	0.92   	290
       	7   	0.95  	0.90  	0.92   	279
       	8   	0.98  	0.88  	0.92   	181
       	9   	0.85  	0.89  	0.87   	176
      	10   	0.93  	0.78  	0.85   	164
      	11   	1.00  	0.53  	0.70    	90
      	12   	0.63  	0.95  	0.76   	335
      	13   	1.00  	0.78  	0.87   	169
      	14   	0.49  	0.96  	0.65   	237
      	15   	1.00  	0.59  	0.74    	46
      	16   	0.86  	0.90  	0.88   	377
      	17   	0.91  	0.94  	0.92   	293
      	18   	0.99  	0.84  	0.91   	232
      	19   	0.89  	0.90  	0.89   	214
      	20   	0.97  	0.82  	0.89   	230
      	21   	0.75  	0.95  	0.84   	382
      	22   	0.79  	0.89  	0.84   	222
      	23   	0.91  	0.85  	0.88   	140
      	24   	1.00  	0.59  	0.74   	145
      	25   	0.98  	0.89  	0.93   	137
      	26   	0.92  	0.92  	0.92   	317
      	27   	0.93  	0.90  	0.92   	237
      	28   	0.92  	0.95  	0.93   	266
      	29   	0.65  	0.93  	0.77   	359
      	30   	0.90  	0.90  	0.90   	237
      	31   	0.96  	0.86  	0.90   	229
      	32   	0.94  	0.92  	0.93   	269
      	33   	0.93  	0.93  	0.93   	306
      	34   	1.00  	0.65  	0.79   	139
      	35   	0.96  	0.84  	0.89   	218
      	36   	0.99  	0.85  	0.91   	220
      	37   	0.98  	0.85  	0.91   	251
      	38   	0.90  	0.80  	0.84   	214
      	39   	0.96  	0.84  	0.90   	186
      	40   	1.00  	0.86  	0.92   	163
      	41   	0.78  	0.97  	0.87   	235
      	42   	0.98  	0.85  	0.91   	242
      	43   	0.94  	0.91  	0.92   	276
      	44   	1.00  	0.86  	0.92   	125
      	45   	1.00  	0.72  	0.84   	167
      	46   	0.97  	0.77  	0.86   	202

	accuracy                       	0.87 	10309
   macro avg   	0.92  	0.85  	0.87 	10309
weighted avg   	0.90  	0.87  	0.88 	10309
```


```
SVM Test Accuracy: 0.26506024096385544
Classification report:           	precision	recall  f1-score   support

       	0   	0.27  	0.19  	0.22    	53
       	1   	0.14  	0.02  	0.04    	44
       	2   	0.26  	0.12  	0.17    	49
       	3   	0.57  	0.28  	0.37    	29
       	4   	0.67  	0.31  	0.42    	26
       	5   	0.17  	0.12  	0.14    	40
       	6   	0.26  	0.29  	0.27    	63
       	7   	0.14  	0.08  	0.11    	60
       	8   	0.43  	0.23  	0.30    	40
       	9   	0.29  	0.32  	0.30    	38
      	10   	0.00  	0.00  	0.00    	36
      	11   	1.00  	0.05  	0.10    	20
      	12   	0.18  	0.42  	0.25    	72
      	13   	0.67  	0.05  	0.10    	37
      	14   	0.14  	0.44  	0.21    	52
      	15   	0.00  	0.00  	0.00    	10
      	16   	0.20  	0.32  	0.25    	82
      	17   	0.38  	0.38  	0.38    	64
      	18   	0.67  	0.04  	0.08    	50
      	19   	0.41  	0.28  	0.33    	46
      	20   	0.25  	0.06  	0.10    	50
      	21   	0.19  	0.52  	0.28    	83
      	22   	0.27  	0.38  	0.32    	48
      	23   	0.52  	0.45  	0.48    	31
      	24   	1.00  	0.03  	0.06    	32
      	25   	0.87  	0.43  	0.58    	30
      	26   	0.19  	0.25  	0.22    	69
      	27   	0.21  	0.13  	0.16    	52
      	28   	0.24  	0.36  	0.29    	58
      	29   	0.20  	0.47  	0.28    	78
      	30   	0.37  	0.35  	0.36    	52
      	31   	0.40  	0.44  	0.42    	50
      	32   	0.24  	0.40  	0.30    	58
      	33   	0.37  	0.44  	0.40    	66
      	34   	0.00  	0.00  	0.00    	30
      	35   	0.26  	0.11  	0.15    	47
      	36   	0.19  	0.08  	0.12    	48
      	37   	0.45  	0.35  	0.40    	54
      	38   	0.34  	0.34  	0.34    	47
      	39   	0.40  	0.20  	0.27    	40
      	40   	0.50  	0.03  	0.05    	36
      	41   	0.30  	0.51  	0.38    	51
      	42   	0.21  	0.11  	0.15    	53
      	43   	0.41  	0.47  	0.43    	60
      	44   	0.50  	0.15  	0.23    	27
      	45   	0.20  	0.03  	0.05    	36
      	46   	0.54  	0.16  	0.25    	44

	accuracy                       	0.27  	2241
   macro avg   	0.35  	0.24  	0.24  	2241
weighted avg   	0.32  	0.27  	0.25  	2241
```

```
Label Encoding Mapping: {'African Violet (Saintpaulia ionantha)': 0,
 'Aloe Vera': 1,
 'Anthurium (Anthurium andraeanum)': 2,
 'Areca Palm (Dypsis lutescens)': 3,
 'Asparagus Fern (Asparagus setaceus)': 4,
 'Begonia (Begonia spp.)': 5,
 'Bird of Paradise (Strelitzia reginae)': 6,
 'Birds Nest Fern (Asplenium nidus)': 7,
 'Boston Fern (Nephrolepis exaltata)': 8,
 'Calathea': 9,
 'Cast Iron Plant (Aspidistra elatior)': 10,
 'Chinese Money Plant (Pilea peperomioides)': 11,
 'Chinese evergreen (Aglaonema)': 12,
 'Christmas Cactus (Schlumbergera bridgesii)': 13,
 'Chrysanthemum': 14,
 'Ctenanthe': 15,
 'Daffodils (Narcissus spp.)': 16,
 'Dracaena': 17,
 'Dumb Cane (Dieffenbachia spp.)': 18,
 'Elephant Ear (Alocasia spp.)': 19,
 'English Ivy (Hedera helix)': 20,
 'Hyacinth (Hyacinthus orientalis)': 21,
 'Iron Cross begonia (Begonia masoniana)': 22,
 'Jade plant (Crassula ovata)': 23,
 'Kalanchoe': 24,
 'Lilium (Hemerocallis)': 25,
 'Lily of the valley (Convallaria majalis)': 26,
 'Money Tree (Pachira aquatica)': 27,
 'Monstera Deliciosa (Monstera deliciosa)': 28,
 'Orchid': 29,
 'Parlor Palm (Chamaedorea elegans)': 30,
 'Peace lily': 31,
 'Poinsettia (Euphorbia pulcherrima)': 32,
 'Polka Dot Plant (Hypoestes phyllostachya)': 33,
 'Ponytail Palm (Beaucarnea recurvata)': 34,
 'Pothos (Ivy arum)': 35,
 'Prayer Plant (Maranta leuconeura)': 36,
 'Rattlesnake Plant (Calathea lancifolia)': 37,
 'Rubber Plant (Ficus elastica)': 38,
 'Sago Palm (Cycas revoluta)': 39,
 'Schefflera': 40,
 'Snake plant (Sanseviera)': 41,
 'Tradescantia': 42,
 'Tulip': 43,
 'Venus Flytrap': 44,
 'Yucca': 45,
 'ZZ Plant (Zamioculcas zamiifolia)': 46}
```

**3: Answer the questions: Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?**

![Fit](https://github.com/user-attachments/assets/c53b111d-3739-4985-a923-86cada3fb370)


Our initial model had a train accuracy of 99% and a test accuracy of 16%. We then did a bit of hyperparameter tuning of pixels_per_cell, cells_per_block, C values, and type of kernel. We realized that linear kernels seemed to massively overfit, getting a high 90% range in train data with test data accuracy between 15% and 20%. This would put the model squarely in the orange arrow. A low C value did not help with this. We then switched to an rbf kernel. We played around with the various parameters, but most results hovered around the ~87% train accuracy range and ~25% test accuracy range. We initially tried to have a lower C of .1 and .01 to simplify the model but it massively underfit the model (light green arrow) with both train and test accuracy around 3%. We then tried C=.5 which put the model around the dark green arrow and had a 46% train accuracy and 20% test accuracy. Ultimately, we went with a model that settles around the purple arrow. It has the highest test accuracy but a super high train accuracy, which leads us to believe that it is still overfitting.


When experimenting for this milestone, we looked at both decision trees and SVMs. We ultimately decided to go with SVMs because it provided higher accuracy with the simple model that we created. For future models, we may try decision trees. We also may look at various feature extraction methods. We went with Histogram of Oriented Gradients (HOG) feature extraction on the recommendation of ChatGPT but may look into other ways to extract features from images for our input feature matrix. 

**4: Update your README.md to include your new work and updates you have all added. Make sure to upload all code and notebooks. Provide links in your README.md**

All code for MS4 is in [MS4-with-reports-maps.ipynb](https://github.com/brentonmdunn/fa24-cse151a-project/blob/Milestone4/MS4-with-reports-maps.ipynb). All of the work for our current model is in the block called Model 2.4. There was an issue in MS3 where the GitHub preview was not rendering the whole file. If you download the ipynb, it should all be there.

The main new work that we did was add an additional model, which for this milestone ended up being an SVM with a rbf kernel. We used the same splits and pre processing as previous milestones.

**5. Conclusion section: What is the conclusion of your 2nd model? What can be done to possibly improve it? Note: The conclusion section should be it's own independent section. i.e. Methods: will have models 1 and 2 methods, Conclusion: will have models 1 and 2 results and discussion.**

Model 1 (CNN):

Boston Fern is predicted correctly really often with the highest recall. This may be because the texture of the fern is very distinct and the CNN is able to detect it well. Similar to daffodils, and hyacinth which are also predicted moderately correctly since they have flowers are easier to identify.
However there were also many that were predicted incorrectly all the time like aloe vera, ctenanthe, and dracaena which be because these plants are primarily long leaved plants which are very common, so specifics with overlapping characteristics are getting confused

There are less images for aloe vera and dracena, abt 100 less images. Since this may be a common issue among some of the underrepresented classes, we can try more data augmentation to increase the diversity


Model 2 (HOG+SVM):

Based on the confusion matrix, the model shows strong performance in correctly predicting some classes, as indicated by the intense blue shades along the diagonal for categories like Chinese Money Plant (Pilea Peperomiodes), Dumb Cane, and Monstera Deliciosa. These classes consistently achieve high prediction accuracy, suggesting that their features are distinct and easily separable by the model. This may be due to unique attributes such as texture, color, or shape that make these classes less prone to overlap with others.
However, the model struggles with several classes that exhibit higher misclassification rates, such as Chrysanthemum, Schefflera, and Tradescantia. These categories often overlap with other classes in the confusion matrix, suggesting that their features may be more ambiguous or similar to those of other plants. For example, Chrysanthemum is frequently confused with Daffodils (Narcissus spp.), possibly due to similarities in floral structure or color patterns. 
A key pattern in the incorrect predictions is that misclassifications frequently occur between plants with similar morphological or visual characteristics. This trend indicates that the model might rely heavily on surface-level visual features and lacks a deeper ability to distinguish subtle differences. Additionally, some classes may suffer from class imbalance in the training data, leading to lower representation and weaker learning for those categories.
To improve the model, it may be helpful to augment the dataset with more diverse and representative images for the misclassified classes. Feature engineering or transfer learning from a pre-trained model specialized in plant identification could also enhance the model’s ability to capture finer distinctions. Finally, refining preprocessing techniques, such as normalizing image backgrounds or focusing on critical regions like flowers or leaves, could help mitigate the observed challenges.


**6. Provide predictions of correct and FP and FN from your test dataset.**

Please review the respective classification report and heatmap for this new model (called HOG+SVM). In addition, you can see the same details for our first model (CNN).

<hr>

</details>


# Written Report

Some exploration code in our notebooks prevented GitHub from showing the notebook preview. If you would like to see the full exploration code, uncomment lines that say `Removed so GitHub preview works`.

Code for MS3: [MS3.ipynb](https://github.com/brentonmdunn/fa24-cse151a-project/blob/Milestone5/MS3.ipynb)

Code for MS4: [MS4-with-reports-maps.ipynb](https://github.com/brentonmdunn/fa24-cse151a-project/blob/Milestone5/MS4-with-reports-maps.ipynb)

## Introduction
At a quick glance, humans are able to distinguish species of plants based on their color, texture, and structure. With knowledge of the differences between plant species, human classification of plants is fairly accurate. It is crucial to correctly identify a plant’s species in order to properly care for the plant. A misdiagnosis can lead to poor plant health or even plant death. 

An efficient, accurate image classification model would be a valuable tool to the general plant-owner, botanist, and anyone who is interested in taking care of or learning more about plants. This would allow for quick identification of a plant species, which could be beneficial to businesses and for education.

Modern image classification methods, like convolutional neural networks (CNN) and support vector machines (SVM), utilize algorithms to efficiently group images based on distinct features. CNNs, modeled after the human brain, involve using connected convolutional and pooling layers in order to identify patterns and features associated with classes of images. These identifiers get passed along between layers, and thus the model is able to effectively train accordingly. SVMs organize data into a hyperplane, dividing points into different classes with the intention of finding the hyperplane that maximizes the margin between the hyperplane and support vectors. While pulling raw image data may be useful, methods have been developed to extract features from raw image data. One such method is called histogram of oriented gradients (HOG) which looks at smaller localized regions and computes a histogram of gradient directions within each cell. 

In this project, we will compare the accuracy between different models - CNN and HOG + SVM - and the tradeoffs that are associated with their differentiations in simplicity, speed, and training techniques. Our model is not entirely accurate when it comes to deciphering more subtle features or underrepresented species. Though it is able to correctly identify some species, enhancements such as data augmentation and refining preprocessing techniques may lead to further improvements in our model.

## Methods
### Data Exploration
The dataset used in this study was sourced from Kaggle and contained a total of 14,790 images of various houseplant species, categorized across 47 different classes. The images were sourced from Bing Images and were manually curated by a non-professional biologist. The dataset was organized in folders, with each folder named after a specific plant species, which made it easy to identify and segregate the data based on plant type.

The images exhibited significant variability in terms of quality, resolution, and size. They were taken in both indoor and outdoor settings and included a variety of perspectives, such as close-up images of plant features and full depictions of entire plants. This variability in image characteristics required careful consideration during preprocessing to ensure consistent and high-quality input for model training.

![image](https://github.com/user-attachments/assets/a0e74172-ac9c-4e7f-8f12-7623beba7615)

**Figure 1:** Bar chart showing number of inputs and their labels

### Preprocessing
First, we checked for and removed any corrupted images to avoid potential issues during training, as corrupted images could negatively affect the model’s performance. Next, we resized all images to a consistent size of 224x224 pixels to ensure uniform input dimensions, which was crucial for the model to process the images efficiently and consistently. Initially, we were going to make the images black and white. However, with the feedback of the TAs, we decided to go with images with color since color is important to distinguish different plants.

Additionally, we created label encoding and one-hot encoding for the 47 plant classes. One-hot encoding was applied for non-ordinal categories, as it is widely used in classification tasks, though it can be computationally expensive and may lead to the curse of dimensionality. On the other hand, label encoding was more efficient but could potentially create unintended ordering of the classes. We experimented with both types of encoding to determine which approach best suited our specific use case. We ended up going with one hot encoding. 

While trying an SVM for MS4, we realized that using raw image data was not getting the most accurate results. We decided to use histogram of oriented gradients (HOG) feature extraction in order to gain more information than just the raw pixel values, which detects the gradient and orientations of different regions. We then scaled the HOG features to normalize the values, since SVMs are highly sensitive to variance in distance. 


### Model 1
We chose a Convolutional Neural Network (CNN) as our first model. Our CNN model ```PlantClassifierCNN``` consists of two convolutional layers (```conv1```, ```conv2```), each followed by a ReLU activation function and max pooling. The convolutional layers use 3x3 kernels with ```stride=1``` and ```padding=1```. ```conv1``` has 3 input channels (for RGB images) and 32 output channels, while ```conv2``` has 32 input channels and 64 output channels. Max pooling uses a 2x2 kernel with ```stride=2```.

The output of the convolutional layers is flattened and passed through two fully connected layers (```fc1```, ```fc2```).  ```fc1``` has an input size of 64 * 56 * 56 and 128 output neurons, followed by a ReLU activation. ```fc2``` has 128 inputs and outputs a number of classes specified in the model initialization. The model uses the Adam optimizer with ```learning rate=0.001``` and CrossEntropyLoss as the loss function. Training was performed for 5 epochs.

At this time, we did not have a good understanding of hyperparameter tuning due to not learning it in the homeworks yet. We experimented with various batch sizes (32, 64, 128) and epochs (1-5).

### Model 2
We chose a Support Vector Machine (SVM) as our second model, with Histogram of Oriented Gradients (HOG) as our feature representation method. Our HOG feature extraction process uses 12 gradient orientations, a 16x16 pixel cell size, and 2x2 cells per block, with ‘L2-Hys' block normalization.

Our SVM model employs an 'rbf' kernel and utilizes default parameters for the SVC class from scikit-learn, and data scaling performed by StandardScaler prior to training.

For this milestone, we did end up experimenting with hyperparameters. We tried both a linear and rbf kernel, different regularization parameters, and various parameters for HOG feature extraction. We ultimately ended up with the following parameters:

- HOG extraction
  - `orientation=12`
  - `pixels_per_cell=(16,16)`
  - `cells_per_block=(2,2)`
  - `block_norm=’L2-Hys’`
- SVC
  - `C=1`
  - `kernel=rbf`

## Results
### Data Exploration
As shown above in the Methods (link to the Methods section) section as well, our dataset consisted of a total of 14,790 images of various houseplant species, with 47 different class categories. These classes are organized in folders, with each folder named after a specific plant species, so it’s easy to identify and segregate the data based on plant type. We also mentioned how the images varied in quality, resolution, and size. In addition, they were taken in various settings and from different perspectives, such as close-up and full depictions. Because of the variations, preprocessing was necessary to ensure consistency. 

### Preprocessing
After performing our preprocessing steps, we resulted in a consistent dataset, where the images are all the same size and quality. Below is an example of preprocessing on an image:
<br>
![image](https://github.com/user-attachments/assets/da85a928-a11b-4d32-ade7-6dff7a24a90e)

**Figure 2:** Image before and after basic pre processing

<br>
After the initial preprocessing, we added extra preprocessing steps for our HOG + SVM model. This included extracting HOG features from the image inputs to capture the plant shapes and appearances by detecting the gradient and orientations of different regions. Then we also scaled the HOG features to normalize the values so that the SVM would not be impacted by specific distances too much especially since SVM is a distance-based classifier

### Train, Validation, and Test Accuracy

Model 1: CNN
- Train accuracy: 76.41%
- Validation accuracy: 19.15%
- Test accuracy: 20.17%

Model 2: HOG + SVM
- Train accuracy: 87.35%
- Validation accuracy: 25.31%
- Test accuracy: 24.50%

### Model 1 Results
For Model 1 **(CNN)**, the training accuracy was 76.41%, but the validation accuracy dropped significantly to 19.15%, with the test accuracy further decreasing to 20.17%. This indicates that the CNN model struggled to generalize to unseen data.
<br>
![image](https://github.com/user-attachments/assets/0f06c556-aa20-495d-aa41-ff061334c6cd)

**Figure 3**: Accuracy graph for train and test data for model 1

<br>
This can be seen from the trend of training and test accuracy from this graph over multiple epochs. Even if we increased the number of epochs, the CNN already stopped improving the accuracy for the testing data.

Model 1 Confusion Matrix:
<br>
<img src="https://github.com/user-attachments/assets/45d270b7-c547-4516-b680-abffb19f19d9" alt="Fit" width="600" />

**Figure 4:** Confusion matrix for model 1

<br>
The dark spots on the diagonal indicate which classes the CNN model predicted more accurately (e.g. Lilium, Boston Fern)
<br><br>

### Model 2 Results
Model 2 **(HOG + SVM)** showed improved performance across all metrics. The training accuracy was 87.35%, with a validation accuracy of 25.31% and a test accuracy of 24.50%. These results indicate a notable improvement in generalization compared to the CNN model, as the drop in accuracy from training to validation and test sets was less pronounced.
<br>
<img src="https://github.com/user-attachments/assets/7b3b47c1-5fd5-4475-aee8-994115c5ed7b" alt="Fit" width="600" />

**Figure 5:** Model performance bar chart for model 2

<br>

- Original: this had the original parameters that we tested with
- Iteration 2: Improvements to the orientation
- Iteration 3: Further improvements to the orientation
- Iteration 4: Reverted to original parameters, but improved the cells per block
- Iteration 5: Reverted to original parameters, but improved the pixels per cells
- Iteration 6: Reverted to original parameters, but changed kernel to rbf instead of linear
- Final: Used best orientation from iteration 3, cells per block from iteration 4, and rbf kernel

Our final is the best since we took the optimized parameters from when the model was using the linear kernel besides the pixels per cell since it resulted in an overfit model. Changing the regularization after these iterations resulted in severe overfitting so we left it at 1.

Model 2 Confusion Matrix:
<br>
<img src="https://github.com/user-attachments/assets/ca07dc53-7bbe-4d5e-96a1-c8ab8ee113d4" alt="Fit" width="600" />

**Figure 6:** Confusion matrix for model 2

<br>
Dark spots on the diagonal indicate which classes the HOG+SVM model predicted more accurately (e.g. Monstera Deliciosa, Chinese Money Plant)

Precision and Recall for Model 2:
```
SVM Train Accuracy: 0.8716655349694442
Classification report:           	precision	recall  f1-score   support

       	0   	0.91  	0.86  	0.88   	247
       	1   	0.99  	0.87  	0.93   	203
       	2   	0.94  	0.90  	0.92   	228
       	3   	1.00  	0.83  	0.91   	131
       	4   	0.99  	0.78  	0.87   	117
       	5   	0.84  	0.79  	0.82   	186
       	6   	0.88  	0.96  	0.92   	290
       	7   	0.95  	0.90  	0.92   	279
       	8   	0.98  	0.88  	0.92   	181
       	9   	0.85  	0.89  	0.87   	176
      	10   	0.93  	0.78  	0.85   	164
      	11   	1.00  	0.53  	0.70    	90
      	12   	0.63  	0.95  	0.76   	335
      	13   	1.00  	0.78  	0.87   	169
      	14   	0.49  	0.96  	0.65   	237
      	15   	1.00  	0.59  	0.74    	46
      	16   	0.86  	0.90  	0.88   	377
      	17   	0.91  	0.94  	0.92   	293
      	18   	0.99  	0.84  	0.91   	232
      	19   	0.89  	0.90  	0.89   	214
      	20   	0.97  	0.82  	0.89   	230
      	21   	0.75  	0.95  	0.84   	382
      	22   	0.79  	0.89  	0.84   	222
      	23   	0.91  	0.85  	0.88   	140
      	24   	1.00  	0.59  	0.74   	145
      	25   	0.98  	0.89  	0.93   	137
      	26   	0.92  	0.92  	0.92   	317
      	27   	0.93  	0.90  	0.92   	237
      	28   	0.92  	0.95  	0.93   	266
      	29   	0.65  	0.93  	0.77   	359
      	30   	0.90  	0.90  	0.90   	237
      	31   	0.96  	0.86  	0.90   	229
      	32   	0.94  	0.92  	0.93   	269
      	33   	0.93  	0.93  	0.93   	306
      	34   	1.00  	0.65  	0.79   	139
      	35   	0.96  	0.84  	0.89   	218
      	36   	0.99  	0.85  	0.91   	220
      	37   	0.98  	0.85  	0.91   	251
      	38   	0.90  	0.80  	0.84   	214
      	39   	0.96  	0.84  	0.90   	186
      	40   	1.00  	0.86  	0.92   	163
      	41   	0.78  	0.97  	0.87   	235
      	42   	0.98  	0.85  	0.91   	242
      	43   	0.94  	0.91  	0.92   	276
      	44   	1.00  	0.86  	0.92   	125
      	45   	1.00  	0.72  	0.84   	167
      	46   	0.97  	0.77  	0.86   	202

	accuracy                       	0.87 	10309
   macro avg   	0.92  	0.85  	0.87 	10309
weighted avg   	0.90  	0.87  	0.88 	10309
```
**Figure 7:** Table showing training prcision and recall for model 2

```
SVM Test Accuracy: 0.26506024096385544
Classification report:           	precision	recall  f1-score   support

       	0   	0.27  	0.19  	0.22    	53
       	1   	0.14  	0.02  	0.04    	44
       	2   	0.26  	0.12  	0.17    	49
       	3   	0.57  	0.28  	0.37    	29
       	4   	0.67  	0.31  	0.42    	26
       	5   	0.17  	0.12  	0.14    	40
       	6   	0.26  	0.29  	0.27    	63
       	7   	0.14  	0.08  	0.11    	60
       	8   	0.43  	0.23  	0.30    	40
       	9   	0.29  	0.32  	0.30    	38
      	10   	0.00  	0.00  	0.00    	36
      	11   	1.00  	0.05  	0.10    	20
      	12   	0.18  	0.42  	0.25    	72
      	13   	0.67  	0.05  	0.10    	37
      	14   	0.14  	0.44  	0.21    	52
      	15   	0.00  	0.00  	0.00    	10
      	16   	0.20  	0.32  	0.25    	82
      	17   	0.38  	0.38  	0.38    	64
      	18   	0.67  	0.04  	0.08    	50
      	19   	0.41  	0.28  	0.33    	46
      	20   	0.25  	0.06  	0.10    	50
      	21   	0.19  	0.52  	0.28    	83
      	22   	0.27  	0.38  	0.32    	48
      	23   	0.52  	0.45  	0.48    	31
      	24   	1.00  	0.03  	0.06    	32
      	25   	0.87  	0.43  	0.58    	30
      	26   	0.19  	0.25  	0.22    	69
      	27   	0.21  	0.13  	0.16    	52
      	28   	0.24  	0.36  	0.29    	58
      	29   	0.20  	0.47  	0.28    	78
      	30   	0.37  	0.35  	0.36    	52
      	31   	0.40  	0.44  	0.42    	50
      	32   	0.24  	0.40  	0.30    	58
      	33   	0.37  	0.44  	0.40    	66
      	34   	0.00  	0.00  	0.00    	30
      	35   	0.26  	0.11  	0.15    	47
      	36   	0.19  	0.08  	0.12    	48
      	37   	0.45  	0.35  	0.40    	54
      	38   	0.34  	0.34  	0.34    	47
      	39   	0.40  	0.20  	0.27    	40
      	40   	0.50  	0.03  	0.05    	36
      	41   	0.30  	0.51  	0.38    	51
      	42   	0.21  	0.11  	0.15    	53
      	43   	0.41  	0.47  	0.43    	60
      	44   	0.50  	0.15  	0.23    	27
      	45   	0.20  	0.03  	0.05    	36
      	46   	0.54  	0.16  	0.25    	44

	accuracy                       	0.27  	2241
   macro avg   	0.35  	0.24  	0.24  	2241
weighted avg   	0.32  	0.27  	0.25  	2241
```
**Figure 8:** Table showing testing prcision and recall for model 2


## Discussion
This project explored two different approaches for classifying plant images: a Convolutional Neural Network (CNN) and a combination of Histogram of Oriented Gradients (HOG) with a Support Vector Machine (SVM). Here is a walkthrough of our process from start to finish.

### The Dataset
The first challenge was with the dataset itself. Some plant species had more examples than others, which created an imbalance and may have made it harder for the models to learn about the less common species. We did some preprocessing–resizing, normalizing, and data augmentation–but looking back, we could’ve gone further. Because there were so many plant classes, there were certain ones that were overrepresented and others that were underrepresented. Techniques like oversampling the underrepresented species, generating more examples might have helped with the imbalance, or class weighting may have been helpful.

### Preprocessing
Preprocessing images had a lot more to it than we initially thought. We started with the basic resizing and scaling. However, we didn't learn about feature extraction until MS4, in which we decided to use HOG extraction. 

### Model 1: Convolutional Neural Network (CNN)
We wanted to start with a CNN model because it’s a common tool for image classification. But in practice, it didn’t perform as well as we hoped. The training accuracy was decent, but when it came to validation and test data, the model struggled. This was probably a sign of overfitting–it learned the training data too well but couldn’t generalize to new images.

The CNN’s architecture was pretty simple, with just two convolutional layers, and it might not have been deep enough to really pick up on the subtle differences between plant species. Also, we only trained it for five epochs, which wasn’t much time for the model to learn, especially for image data. On top of that, while we did some basic data augmentation, adding more variations like changing lighting or rotation might have helped the CNN handle real-world scenarios better.

### Model 2: HOG + SVM
The HOG + SVM model did better overall, as shown in the figure from the models section, from optimizing the various parameters including the HOG features and kernel.  It did a good job of pulling out clear features like shapes and edges, and the SVM uses those features to separate the plant species into classes. While it outperformed the CNN, it still wasn’t ideal–especially on the validation and test sets. Similarly, this model showed having trouble with generalization too.

One thing about HOG + SVM is that it relies on manually crafted features, which means it might miss some of the more complex patterns that a deep learning model like a CNN could potentially learn. Although we did hyperparameter tuning, there were some parameters that we could have also tested, such as number of bins in HOG or adjusting gamma in SVM parameters.

### Comparing the Two
The CNN had the potential to automatically learn features directly from the images, but it needed more training, a better architecture, and maybe even more data. On the other hand, the HOG + SVM model didn’t require as much training or computational power, so it performed better under these constraints. But it also has its limits–it’s not as flexible and can’t adapt as well to complex datasets.

### Shortcomings
Both of our models overfit to the data. For our CNN in Milestone 3, this was largely due to a limitation in our understanding of CNNs and hyperparameter tuning in general. The overfitting observed was likely due to a lack of regularization - both our CNN and SVM models may have become excessively sensitive to specific patterns from the training data and the accompanying noise. Both these models are capable of representing complex functions, which may have led them to capture meaningless characteristics from our training data.

We were also limited by computation power, particularly for the SVM model. Among our group, it took on average over 2 hours for each iteration, making hyperparameter training difficult due to the amount of time it took for each iteration. We were also unfamiliar with SDSC due to the recording never coming out. If we had access to more compute, we could better hyperparameter tune.

## Conclusion
### What We Learned
Although the test accuracy for both models was lower than we expected, there are a lot of things we have learned for the future:
1. Use techniques like oversampling or generating synthetic data to give the underrepresented species a better proportion.
2. For the CNN, experiment with deeper architectures or use pre-trained models like ResNet or VGG. And, we can also train with more epochs.
3. Perform more hyperparameter tuning on the SVM.
4. Add metrics like precision, recall, and F1-score to see where the models do well and where they fall short.

We tested the pre-trained model ResNet that significantly boosted the accuracy of our model. However, we were unsure if we were allowed to utilize a pre-trained model and thus left it out of our final submission and analysis. If we were to go back and improve our model without any restrictions, we would highly consider using ResNet.

### Future Work 
While brainstorming and researching for what models we could use for milestones, another model type that we were considering were decision trees. This is something we can look at for future models with this dataset. We also learned a lot about hyperparameter tuning and neural networks after doing homeworks 4 and 5, and feel that we could have implemented a better CNN model for our milestone after doing homeworks 4 and 5. 

Overall, this project taught us a lot about the trade-offs between traditional machine learning and deep learning. CNNs are powerful but need careful setup and lots of data to work well. HOG + SVM is a simpler option that works decently in constrained situations but doesn’t have the flexibility of a deep learning model. While neither model was perfect, they each showed potential and gave us ideas for what to try next. This process reinforced the idea that machine learning isn’t about finding the perfect solution–it’s about constant learning, testing, and improving.

## Statement of Collaboration
- Catherine Du  (Coder, Writer, Reviewer): Assisted with preprocessing, tuned HOG + SVM and added conclusions for MS4, added visuals for report
- Brenton Dunn (Lead, Coder): Researched models for MS4, hyperparameter tuned SVM, went to TA office hours for more feedback
- Matthew Tan (Coder, Writer, Reviewer): Documented preprocessing, experimented with potential models, and contributed to methods and results sections of written report
- Trisha Tong (Coder, Writer): Performed data exploration, researched models for MS4, contributed to written report
- Sophia Yu (Coder, Writer) preprocessed data, researched Model 1, graph plotting
