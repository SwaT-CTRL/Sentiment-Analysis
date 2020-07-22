import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pickle

data = []
def user_data(value):
    value1 = re.sub('[^a-zA-Z]', ' ', value)
    value1 = value1.lower()
    value1 = value1.split()
    value1 = [word for word in value1 if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    value1 = [ps.stem(word) for word in value1]
    value1 = ' '.join(value1)
    data.append(value1)
    return data
