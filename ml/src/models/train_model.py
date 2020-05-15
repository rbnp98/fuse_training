import sys
sys.path.insert(1, '/home/prajin/Desktop/Sentimnet Analysis/fuse_training-master/ml/src/features')

from build_features import naive_bayes_bag_of_words, logistic_regression_bag_of_words
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support as score
import mlflow
from sklearn.metrics import accuracy_score

# For calculating Multicclass ROC_AUC_SCORE
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

# Training Naive Bayes - Bag of Words.
def train_naive_bayes_bow():
    final_Xtr_bow, final_Xcv_bow, final_Xtest_bow, y_tr,y_cv,y_test = naive_bayes_bag_of_words()
    auc_train=[]
    auc_cv=[]
    alpha_values = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]

    for i in alpha_values:
        mnb = MultinomialNB(alpha = i)
        mnb.fit(final_Xtr_bow,y_tr)
        pred=mnb.predict(final_Xcv_bow)
        pred1=mnb.predict(final_Xtr_bow)
        auc_train.append(multiclass_roc_auc_score(y_tr,pred1))
        mlflow.end_run()
        with mlflow.start_run() as run:
            mlflow.log_metric('ROC_AUC_SCORE', multiclass_roc_auc_score(y_cv,pred))
            mlflow.log_param('Alpha Value', i)
            mlflow.log_metric('Accuuracy', accuracy_score(y_cv, pred))
            mlflow.log_param('Alhpa Value', i)
        auc_cv.append(multiclass_roc_auc_score(y_cv,pred))
        optimal_alpha= alpha_values[auc_cv.index(max(auc_cv))]
    
    # training with values maximum alpha 
    mnb=MultinomialNB(alpha=optimal_alpha)
    mnb.fit(final_Xtr_bow,y_tr)

    # calculating F1_score , Precision, Recall, Accuracy, Support 
    # Prediction 
    y_train_pred = mnb.predict(final_Xtr_bow)
    y_test_pred  = mnb.predict(final_Xtest_bow)

    # training accuracy
    precision_train_nb_bow, recall_train_nb_bow, fscore_train_nb_bow, support_train_nb_bow = score(y_tr, y_train_pred)

    # Testing Accuracy
    precision_test_nb_bow, recall_test_nb_bow, fscore_test_nb_bow, support_test_nb_bow = score(y_test, y_test_pred)

    # ROC_AUC_SCORE
    test_roc_auc_nb_bow =  multiclass_roc_auc_score(y_test, y_test_pred)
    train_roc_auc_nb_bow = multiclass_roc_auc_score(y_tr, y_train_pred)

    return y_tr, y_train_pred, y_test, y_test_pred


def train_log_reg_bow():
    X_train_bow, X_cv_bow, X_test_bow, y_tr, y_cv, y_test = logistic_regression_bag_of_words()
    C = [10**-3, 10**-2, 10**0, 10**2,10**3,10**4]
    auc_train=[]
    auc_cv=[]
    for c in C:
        lr=LogisticRegression(penalty='l1',C=c)
        lr.fit(X_train_bow,y_tr)
        cv=lr.predict(X_cv_bow)
        with mlflow.start_run() as run:
            mlflow.log_metric('ROC_AUC_SCORE', multiclass_roc_auc_score(y_cv,cv))
            mlflow.log_param('Alpha Value', i)
            mlflow.log_metric('Accuuracy', accuracy_score(y_cv, cv))
            mlflow.log_param('Alhpa Value', i)
        auc_cv.append(multiclass_roc_auc_score(y_cv,cv))
        tr=lr.predict(X_train_bow)
        auc_train.append(multiclass_roc_auc_score(y_tr,tr))

    optimal_c= C[auc_cv.index(max(auc_cv))]
    log_reg=LogisticRegression(C=optimal_c)
    log_reg.fit(X_train,y_train)

    y_train_pred = log_reg.predict(X_train_bow)
    y_test_pred  = log_reg.predict(X_test_bow)

    # accuracy - training
    precision_train__log_bow, recall_train_log_bow, fscore_train_log_bow, support_train_log_bow = score(y_tr, y_train_pred)

    #testing
    precision_test__log_bow, recall_test_log_bow, fscore_test_log_bow, support_test_log_bow = score(y_test, y_test_pred)

    #ROC_AUC_SCORE
    test_roc_auc_log_bow =  multiclass_roc_auc_score(y_test, y_test_pred)
    train_roc_auc_log_bow = multiclass_roc_auc_score(y_tr, y_train_pred)

    return y_tr, y_train_pred, y_test, y_test_pred


    














