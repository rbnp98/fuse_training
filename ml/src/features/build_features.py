import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import StandardScaler

sys.path.insert(1, '/home/prajin/Desktop/Sentimnet Analysis/fuse_training-master/ml/src/data')

from make_dataset import make_data


dataset , dataset_corpus = make_data()

X = dataset_corpus
y = np.array(dataset['emotion'])

X_1, X_test, y_1, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3)

def naive_bayes_bag_of_words():
    count_vect=CountVectorizer()
    final_Xtr_bow=count_vect.fit_transform(X_tr)
    final_Xcv_bow=count_vect.transform(X_cv)
    final_Xtest_bow=count_vect.transform(X_test)

    return final_Xtr_bow, final_Xcv_bow, final_Xtest_bow, y_tr,y_cv,y_test


def logistic_regression_bag_of_words():
    count_vect = CountVectorizer()
    X_train_bow=count_vect.fit_transform(X_tr)
    X_cv_bow =  count_vect.transform(X_cv)
    X_test_bow=count_vect.transform(X_test)

    scalar = StandardScaler(with_mean=False)
    X_train_bow = scalar.fit_transform(X_train)
    X_test_bow= scalar.transform(X_test)
    X_cv_bow=scalar.transform(X_cv)

    return X_train_bow, X_cv_bow, X_test_bow, y_tr, y_cv, y_test






