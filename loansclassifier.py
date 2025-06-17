# Load libraries
from numpy import array
from pandas import DataFrame
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from seaborn import set_style
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Load dataset
loan_data = pd.read_csv('loan_approval_dataset.csv')
loan_data.head()
#Structure of the dataset
loan_data.shape
#Missing values
loan_data.isnull().sum()
#
loan_data.describe()
