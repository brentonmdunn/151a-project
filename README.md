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

## Preprocessing Questions
Where does your model fit in the fitting graph? What are the next models you are thinking of and why?

We are possibly underfitting since our:
- Training data is around 0.2900 
- Validation is around 0.1866
- Test is around 0.1945
- Baseline loss would be around 3.85, and since our loss reach 2.644 we are improving but ideally loss would be closer to 1
- Even though the training dataset is higher in accuracy than the validation and test, holistically speaking, we would need to run the epochs more to capture the data better.
  ![image](https://github.com/user-attachments/assets/477c517e-4d5f-4e15-86f4-43bb1601cdd2)

We could try:
- Deeper CNN model since the additional layers may be able to catch smaller details like certain patterns on the leaves
- CNN with attention so that it could focus on more important parts, also like the patterns on leaves.
- Pretrained models with transfer learning since it can reduce the training time since our model takes a while to train, and improve accuracy at the same time.

## Preprocessing Conclusion
What is the conclusion of your 1st model? What can be done to possibly improve it?

The conclusion of our 1st model is:
- It has an accuracy higher than random since random would only be only be 2.12%, however it is overfitting to the training data as mentioned above with the preprocessing questions

We could improve it by:
- Increasing the number of epochs
- Add data augmentation by adding transformations (torchvision.transforms, flips, scaling)
- Change learning rate
- Regularization (dropout, weight decay)
- Change network architecture parameters


# Ground Truth and Example Predictions for Train, Validation, and Test
![image](https://github.com/user-attachments/assets/551d307a-5f93-4fb8-8472-5046451755e3)

