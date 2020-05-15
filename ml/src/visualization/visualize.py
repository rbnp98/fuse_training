import sys
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/prajin/Desktop/Sentimnet Analysis/fuse_training-master/ml/src/models')

from train_model import train_naive_bayes_bow
from train_model import train_log_reg_bow

def confusion_matrix_train_test_nb_bow():
    y_tr, y_train_pred, y_test, y_test_pred = train_naive_bayes_bow()
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_tr, y_train_pred)
    class_label = ["joy", "sadness", "anger" ,"fear", "shame", "disgust", "guilt"]
    df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
    sns.heatmap(df, annot = True,fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    
    conf_mat = confusion_matrix(y_test, y_test_pred)
    df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
    sns.heatmap(df, annot = True,fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
