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


(2) Or download from [Kaggle](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species).


Move the data (either with `mv` or with GUI) so that the file structure looks like the following:
- fa24-cse151a-project (current working directory)
  - house_plant_species
  - download_data.py
  - preprocessing.ipynb
  - requirements.txt


<hr>


Colab link: https://colab.research.google.com/drive/1Inr1L6lXyXvhoxw_a7Zp9FFUmnFG9YcH?usp=sharing
