**[Milestone 4 bookmark](https://github.com/brentonmdunn/fa24-cse151a-project/tree/Milestone4?tab=readme-ov-file#ms4)**

# CSE 151A Project

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

TODO

**6. Provide predictions of correct and FP and FN from your test dataset.**

Please review the respective classification report and heatmap for this new model (called HOG+SVM). In addition, you can see the same details for our first model (CNN).
