import pandas
#import data
#data = pandas.read_csv('data.csv')

y = data['project_is_approved'].values
X = data.drop(['project_is_approved'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)

# BOW
from sklearn.feature_extraction.text import CountVectorizer

print("="*100)

#min of 10 times occurred word, and range 1 is min max of 4 - range
bow_vectorizer = CountVectorizer(min_df=10,ngram_range=(1,4), max_features=5000)
bow_vectorizer.fit(X_train['essay'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_essay_bow = bow_vectorizer.transform(X_train['essay'].values)
X_cv_essay_bow = bow_vectorizer.transform(X_cv['essay'].values)
X_test_essay_bow = bow_vectorizer.transform(X_test['essay'].values)

print("After vectorizations")
print(X_train_essay_bow.shape, y_train.shape)
print(X_cv_essay_bow.shape, y_cv.shape)
print(X_test_essay_bow.shape, y_test.shape)
print("="*100)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=10)

tfidf_vectorizer.fit(X_train['essay'].values)
print("Set 2 - TFIDF vecotorization")

X_train_essay_tfidf = tfidf_vectorizer.transform(X_train['essay'].values)
X_cv_essay_tfidf = tfidf_vectorizer.transform(X_cv['essay'].values)
X_test_essay_tfidf = tfidf_vectorizer.transform(X_test['essay'].values)

print("After vectorizations")
print(X_train_essay_tfidf.shape, y_train.shape)
print(X_cv_essay_tfidf.shape, y_cv.shape)
print(X_test_essay_tfidf.shape, y_test.shape)
print("="*100)

school_state_vectorizer = CountVectorizer()
school_state_vectorizer.fit(X_train['school_state'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_state_ohe = school_state_vectorizer.transform(X_train['school_state'].values)
X_cv_state_ohe = school_state_vectorizer.transform(X_cv['school_state'].values)
X_test_state_ohe = school_state_vectorizer.transform(X_test['school_state'].values)

print("After vectorizations")
print(X_train_state_ohe.shape, y_train.shape)
print(X_cv_state_ohe.shape, y_cv.shape)
print(X_test_state_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)


teacher_prefix_vectorizer = CountVectorizer()
teacher_prefix_vectorizer.fit(X_train['teacher_prefix'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_teacher_ohe = teacher_prefix_vectorizer.transform(X_train['teacher_prefix'].values)
X_cv_teacher_ohe = teacher_prefix_vectorizer.transform(X_cv['teacher_prefix'].values)
X_test_teacher_ohe = teacher_prefix_vectorizer.transform(X_test['teacher_prefix'].values)

print("After vectorizations")
print(X_train_teacher_ohe.shape, y_train.shape)
print(X_cv_teacher_ohe.shape, y_cv.shape)
print(X_test_teacher_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)


project_grade_category_vectorizer = CountVectorizer()
project_grade_category_vectorizer.fit(X_train['project_grade_category'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_grade_ohe = project_grade_category_vectorizer.transform(X_train['project_grade_category'].values)
X_cv_grade_ohe = project_grade_category_vectorizer.transform(X_cv['project_grade_category'].values)
X_test_grade_ohe = project_grade_category_vectorizer.transform(X_test['project_grade_category'].values)

print("After vectorizations")
print(X_train_grade_ohe.shape, y_train.shape)
print(X_cv_grade_ohe.shape, y_cv.shape)
print(X_test_grade_ohe.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)


from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
# normalizer.fit(X_train['price'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['price'].values.reshape(1,-1))

X_train_price_norm = normalizer.transform(X_train['price'].values.reshape(1,-1))
X_cv_price_norm = normalizer.transform(X_cv['price'].values.reshape(1,-1))
X_test_price_norm = normalizer.transform(X_test['price'].values.reshape(1,-1))

X_train_price_norm = X_train_price_norm.reshape(-1,1)
X_cv_price_norm = X_cv_price_norm.reshape(-1,1)
X_test_price_norm = X_test_price_norm.reshape(-1,1)

print("After vectorizations")
print(X_train_price_norm.shape, y_train.shape)
print(X_cv_price_norm.shape, y_cv.shape)
print(X_test_price_norm.shape, y_test.shape)
print("="*100)

clean_categories_vectorizer = CountVectorizer()
clean_categories_vectorizer.fit(X_train['clean_categories'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_cate = clean_categories_vectorizer.transform(X_train['clean_categories'].values)
X_cv_cate = clean_categories_vectorizer.transform(X_cv['clean_categories'].values)
X_test_cate = clean_categories_vectorizer.transform(X_test['clean_categories'].values)

print("After vectorizations")
print(X_train_cate.shape, y_train.shape)
print(X_cv_cate.shape, y_cv.shape)
print(X_test_cate.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)


clean_subcategories_vectorizer = CountVectorizer()
clean_subcategories_vectorizer.fit(X_train['clean_subcategories'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_sub_cate = clean_subcategories_vectorizer.transform(X_train['clean_subcategories'].values)
X_cv_sub_cate = clean_subcategories_vectorizer.transform(X_cv['clean_subcategories'].values)
X_test_sub_cate = clean_subcategories_vectorizer.transform(X_test['clean_subcategories'].values)

print("After vectorizations")
print(X_train_sub_cate.shape, y_train.shape)
print(X_cv_sub_cate.shape, y_cv.shape)
print(X_test_sub_cate.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)


normalizer = Normalizer()
# normalizer.fit(X_train['price'].values)
# this will rise an error Expected 2D array, got 1D array instead: 
# array=[105.22 215.96  96.01 ... 368.98  80.53 709.67].
# Reshape your data either using 
# array.reshape(-1, 1) if your data has a single feature 
# array.reshape(1, -1)  if it contains a single sample.
normalizer.fit(X_train['teacher_number_of_previously_posted_projects'].values.reshape(1,-1))

X_train_proj_count_norm = normalizer.transform(X_train['teacher_number_of_previously_posted_projects'].values.reshape(1,-1))
X_cv_proj_count_norm = normalizer.transform(X_cv['teacher_number_of_previously_posted_projects'].values.reshape(1,-1))
X_test_proj_count_norm = normalizer.transform(X_test['teacher_number_of_previously_posted_projects'].values.reshape(1,-1))

X_train_proj_count_norm = X_train_proj_count_norm.reshape(-1,1)
X_cv_proj_count_norm = X_cv_proj_count_norm.reshape(-1,1)
X_test_proj_count_norm = X_test_proj_count_norm.reshape(-1,1)

print("After vectorizations")
print(X_train_proj_count_norm.shape, y_train.shape)
print(X_cv_proj_count_norm.shape, y_cv.shape)
print(X_test_proj_count_norm.shape, y_test.shape)
print("="*100)

# Bag of words set
from scipy.sparse import hstack
X_tr_bow = hstack((X_train_essay_bow, X_train_state_ohe, X_train_teacher_ohe, X_train_grade_ohe, X_train_cate, X_train_sub_cate,X_train_price_norm, X_train_proj_count_norm)).tocsr()
X_cv_bow = hstack((X_cv_essay_bow, X_cv_state_ohe, X_cv_teacher_ohe, X_cv_grade_ohe, X_cv_cate, X_cv_sub_cate,X_cv_price_norm, X_cv_proj_count_norm)).tocsr()
X_te_bow = hstack((X_test_essay_bow, X_test_state_ohe, X_test_teacher_ohe, X_test_grade_ohe, X_test_cate, X_test_sub_cate,X_test_price_norm, X_test_proj_count_norm)).tocsr()

print("Final Data matrix")
print(X_tr_bow.shape, y_train.shape)
print(X_cr_bow.shape, y_cv.shape)
print(X_te_bow.shape, y_test.shape)
print("="*100)

# tfidf set
X_tr_tfidf = hstack((X_train_essay_tfidf, X_train_state_ohe, X_train_teacher_ohe, X_train_grade_ohe, X_train_cate, X_train_sub_cate,X_train_price_norm, X_train_proj_count_norm)).tocsr()
X_cv_tfidf = hstack((X_cv_essay_tfidf, X_cv_state_ohe, X_cv_teacher_ohe, X_cv_grade_ohe, X_cv_cate, X_cv_sub_cate,X_cv_price_norm, X_cv_proj_count_norm)).tocsr()
X_te_tfidf = hstack((X_test_essay_tfidf, X_test_state_ohe, X_test_teacher_ohe, X_test_grade_ohe, X_test_cate, X_test_sub_cate,X_test_price_norm, X_test_proj_count_norm)).tocsr()

print("Final Data matrix")
print(X_tr_tfidf.shape, y_train.shape)
print(X_cr_tfidf.shape, y_cv.shape)
print(X_te_tfidf.shape, y_test.shape)
print("="*100)


# Perform MNB
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def TrainAndPredict(alpha_values, train_data, cv_data):
    train_auc = []
    y_cv_predi = []
    for a in alpha_values:
        clf = MultinomialNB(alpha=a)
        clf.fit(train_data, y_train)
        
        y_train_pred = clf.predict(train_data)
        y_cv_pred = clf.predict(cv_data)
        
        train_score = roc_auc_score(y_train,y_train_pred)
        print("="*50)
        print("Alpha - ",a)
        print("Training score - ",train_score)
        cv_score = roc_auc_score(y_cv, y_cv_pred)
        print("CV score- ", cv_score)
        print("="*50)
        train_auc.append(train_score)
        y_cv_predi.append(cv_score)
    return train_auc, y_cv_predi


alpha_values = [0.0001,0.001,0.01,0.1,1,10,100,1000]
print("Apply NB for BOW")
train_auc, cv_auc = TrainAndPredict(alpha_values, X_tr_bow, X_cv_bow)
print("="*50)


import matplotlib.pyplot as plt

def PlotAucCurve(alpha_values,train_auc,cv_auc):
    plt.plot(alpha_values, train_auc, label='Train AUC')
    plt.plot(alpha_values, cv_auc, label='CV AUC')

    plt.scatter(alpha_values, train_auc, label='Train AUC points')
    plt.scatter(alpha_values, cv_auc, label='CV AUC points')

    plt.legend()
    plt.xlabel("Alpha: hyperparameter")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.grid()
    plt.show()

PlotAucCurve(alpha_values,train_auc,cv_auc)
train_auc.sort()
print(train_auc)
cv_auc.sort()
print(cv_auc)


print("Apply NB for TFIDF")
train_auc_idf, cv_auc_idf = TrainAndPredict(alpha_values, X_tr_tfidf, X_cv_tfidf)
print("="*50)


PlotAucCurve(alpha_values,train_auc_idf, cv_auc_idf)
train_auc_idf.sort()
print(train_auc_idf)
cv_auc_idf.sort()
print(cv_auc_idf)

from sklearn.metrics import roc_curve, auc
import numpy as np

def find_best_threshold(threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    return t

def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


def TrainAndPlotNB(alpha, train, test):
    clf = MultinomialNB(alpha=alpha)
    clf.fit(train, y_train)

    y_train_pred = clf.predict(train)
    y_test_pred = clf.predict(test)

    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.legend()
    plt.xlabel("Alpha: hyperparameter")
    plt.ylabel("AUC")
    plt.grid()
    return plt,train_fpr, train_tpr, tr_thresholds,test_fpr, test_tpr, te_thresholds,y_train_pred, y_test_pred 

plt,train_fpr, train_tpr, tr_thresholds,test_fpr, test_tpr, te_thresholds, y_train_pred, y_test_pred  = TrainAndPlotNB(0.1, X_tr_bow, X_te_bow)
plt.title("ERROR PLOTS - BOW with alpha - 0.1")
plt.show()

print("="*100)
from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)
print("Train confusion matrix")
print(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)))
print("Test confusion matrix")
print(confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)))



plt,train_fpr, train_tpr, tr_thresholds,test_fpr, test_tpr, te_thresholds, y_train_pred, y_test_pred  = TrainAndPlotNB(0.1, X_tr_tfidf, X_te_tfidf)
plt.title("ERROR PLOTS TFIDF with alpha - 0.1")
plt.show()

print("="*100)
from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)
print("Train confusion matrix")
print(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)))
print("Test confusion matrix")
print(confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)))


plt,train_fpr, train_tpr, tr_thresholds,test_fpr, test_tpr, te_thresholds, y_train_pred, y_test_pred   = TrainAndPlotNB(0.01, X_tr_tfidf, X_te_tfidf)
plt.title("ERROR PLOTS TFIDF with alpha - 0.01")
plt.show()

print("="*100)
from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)
print("Train confusion matrix")
print(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)))
print("Test confusion matrix")
print(confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)))


clf = MultinomialNB(alpha=0.1)
clf.fit(X_tr_bow, y_train)
# https://datascience.stackexchange.com/questions/65219/find-the-top-n-features-from-feature-set-using-absolute-values-of-feature-log-p
sorted_prob_feature_indexes = clf.feature_log_prob_[1, :].argsort()
top_20_feature_index = sorted_prob_feature_indexes[:20]

features_lst_bow = list(bow_vectorizer.get_feature_names() + school_state_vectorizer.get_feature_names() + \
                    teacher_prefix_vectorizer.get_feature_names() + project_grade_category_vectorizer.get_feature_names() + \
                    clean_categories_vectorizer.get_feature_names() + \
                    clean_subcategories_vectorizer.get_feature_names() +  ["Price"] + ["teacher_number_of_previously_posted_projects"])

features_lst_tfidf = list(tfidf_vectorizer.get_feature_names() + school_state_vectorizer.get_feature_names() + \
                    teacher_prefix_vectorizer.get_feature_names() + project_grade_category_vectorizer.get_feature_names() + \
                    clean_categories_vectorizer.get_feature_names() + \
                    clean_subcategories_vectorizer.get_feature_names() +  ["Price"] + ["teacher_number_of_previously_posted_projects"])


def PrintTop20Feature(features_lst,top_20_feature_index):
    for index in top_20_feature_index:
        print(features_lst[index])
print("="*50)
print("Top 20 features for BOW set")
PrintTop20Feature(features_lst,top_20_feature_index)
print("="*50)
print("")
print("="*50)
print("Top 20 features for TFIDF set")
PrintTop20Feature(features_lst,top_20_feature_index)
print("="*50)


from prettytable import PrettyTable
#https://stackoverflow.com/questions/522563/accessing-the-index-in-for-loops

data = [["BOW","NB","0.1","0.6625623400271925"],["TFIDF","NB","0.01","0.5100275604879596"]]
df = pandas.DataFrame(data, columns=['Vectorizer','Model', 'HyperParameter', 'AUC'])

def generate_ascii_table(df):
    x = PrettyTable()
    x.field_names = df.columns.tolist()
    for row in df.values:
        x.add_row(row)
    print(x)
    return x

generate_ascii_table(df)