# Import Libraries
import pandas as pd
import numpy as np
import string, nltk
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.models import Model, Sequential
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

punct = nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
nltk.download('omw-1.4')
lemma = WordNetLemmatizer()
stemm = PorterStemmer()
tokenize = WordPunctTokenizer()