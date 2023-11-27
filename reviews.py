import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

data_file_path = 'train.txt'
data = pd.read_csv(data_file_path, sep='\t', header=None, names=['label', 'text'])
data['text'] = data['text'].astype(str)

reviews = data['text']
labels = data['label']

# simple preprocessing
reviews = reviews.str.lower()
reviews = reviews.str.replace(r'[^\w\s]', '', regex=True)

# splitting into train, validation, test data
X_train, X_val, y_train, y_val = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# TF-ID vectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# # Support Vector classifier
svm_classifier = SVC()

param_grid = {
    'C': [2, 3, 4],
    'gamma': [.4, .45, .5],  # For RBF kernel
}

# best parameter search
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, scoring='accuracy')

#training
grid_search.fit(X_train_tfidf, y_train)

best_svm_classifier = grid_search.best_estimator_

# fitting on eval data
y_val_pred = best_svm_classifier.predict(X_val_tfidf)

print("Classification report: ")
print(classification_report(y_val, y_val_pred))

fig, ax = plt.subplots(figsize=(10, 5))
ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, ax=ax)
plt.xticks(rotation=45)
_ = ax.set_title(
    f"Confusion Matrix for {svm_classifier.__class__.__name__}\non the original documents"
)

negative_deceptive = list()
negative_truthful = list()
positive_deceptive = list()
positive_truthful = list()
truthful_negative = list()
truthful_positive = list()
deceptive_negative = list()
deceptive_positive = list()

print("Saving the wrongly classified data:")
for row_index, (input, prediction, label) in enumerate(zip(X_val_tfidf, y_val_pred, y_val)):
    if prediction != label:
        print('Row', row_index, 'has been classified as ', prediction, 'and should be ', label)
        if prediction == "DECEPTIVENEGATIVE" and label == "TRUTHFULNEGATIVE":
            negative_deceptive.append(row_index)
        elif prediction == "TRUTHFULNEGATIVE" and label == "DECEPTIVENEGATIVE":
            negative_truthful.append(row_index)
        elif prediction == "DECEPTIVEPOSITIVE" and label == "TRUTHFULPOSITIVE":
            positive_deceptive.append(row_index)
        elif prediction == "TRUTHFULPOSITIVE" and label == "DECEPTIVEPOSITIVE":
            positive_truthful.append(row_index)
        elif prediction == "TRUTHFULNEGATIVE" and label == "TRUTHFULPOSITIVE":
            truthful_negative.append(row_index)
        elif prediction == "TRUTHFULPOSITIVE" and label == "TRUTHFULNEGATIVE":
            truthful_positive.append(row_index)
        elif prediction == "DECEPTIVENEGATIVE" and label == "DECEPTIVEPOSITIVE":
            deceptive_negative.append(row_index)
        elif prediction == "DECEPTIVEPOSITIVE" and label == "DECEPTIVENEGATIVE":
            deceptive_positive.append(row_index)


print("Checking the individual sentences:\n")
for x in deceptive_positive:
    words = X_val.iloc[x].split()
    words = set(words)
    words = words - ENGLISH_STOP_WORDS
    print(X_val.iloc[x])
    print(words)

# dooing the same for the test set
test_file_path = 'test_just_reviews.txt'
test_df = pd.read_csv(test_file_path, sep='\t', header=None, names=['text'])
test_df['text'] = test_df['text'].astype(str)
test_reviews = test_df['text']
test_reviews = test_reviews.str.lower()
test_reviews = test_reviews.str.replace(r'[^\w\s]', '', regex=True)
test_reviews_tfidf = tfidf_vectorizer.transform(test_reviews)
test_pred = best_svm_classifier.predict(test_reviews_tfidf)
df_test_pred = pd.DataFrame(test_pred)
df_test_pred[0].to_csv('results.txt', header=None, index=False)
