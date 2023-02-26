import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt



class Color:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"

c = Color()

def launchProgram():
    print(f"{c.CYAN}+++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"{c.CYAN}+              SPAM OR HAM PROJECT                  +")
    print(f"{c.CYAN}+       AML2203 Midterm Project - Group 3           +")
    print(f"{c.CYAN}+++++++++++++++++++++++++++++++++++++++++++++++++++++{c.RESET}\n")

launchProgram()

# Load the dataset
df = pd.read_csv("spam_dataset.csv")
print(f"{c.GREEN} Count of Dataset: {c.RESET}\n {df.count()}\n")
print(f"{c.GREEN} Info of Dataset: {c.RESET}\n {df.info()}\n") #To be checked on why the label is displayed at the bottom of the result of df.info()
print(f"{c.GREEN} Data Frame Shape: {c.RESET}\n {df.shape}\n")
print(f"{c.GREEN} Preview of Dataset: {c.RESET}\n {df.head()}\n")


# Convert the text column into numerical features using the bag-of-words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df["label"], test_size=0.2)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict the label of the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"{c.GREEN} ***Accuracy: {c.RESET} {accuracy}\n")

# Show Prediction Report
# plt.scatter(X_test, y_test, color ='b')
# plt.plot(X_test, y_pred, color ='k')
# plt.show()
# Data scatter of predicted values
# print precision score






def evalModel(model, x_train, x_test, y_train, y_test, y_pred): # This method evaluate the Logistic Regression model and provide clasification report, confusion matrix and precision score

    print(f"{c.GREEN}***Score: {c.RESET} {model.score(x_test, y_test)}\n")

    # Classification Report without cross-validation
    # For reference on classification matrix https://www.simplilearn.com/tutorials/machine-learning-tutorial/confusion-matrix-machine-learning#:~:text=A%20confusion%20matrix%20presents%20a,actual%20values%20of%20a%20classifier.
    print(f"{c.GREEN}***Classification Report*** {c.RESET}")
    print(classification_report(y_test, y_pred))

    # k-fold cross-validation and confusion matrices
    y_train_pred = cross_val_predict(model, x_train, y_train, cv=5)
    print(f"{c.GREEN}***Confusion Matrix*** {c.RESET}")
    print(confusion_matrix(y_train, y_train_pred), "\n")
#    print(f"Precision Score: {precision_score(y_train, y_train_pred)}")




evalModel(model, X_train, X_test, y_train, y_test, y_pred) #This method generates the Classification Report
