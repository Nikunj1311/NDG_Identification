import pandas as pd
from termcolor import colored as cl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC

ATTRS = ["bold"]
COLOR = "blue"

df = pd.read_csv("parkinsons.csv")

y = df['status']
x=df[['MDVP:Fhi(Hz)','spread1','MDVP:Fo(Hz)','MDVP:Flo(Hz)','D2','HNR','MDVP:Shimmer(dB)']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
ros = RandomOverSampler(sampling_strategy="not minority", random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)

models = []  # storing the different models
models.append(("SVM", SVC(kernel="linear")))
my_dict = {}
for name, model in models:
    md = model.fit(X_train, y_train)
    y_pred = md.predict(X_test)
    print(
        cl(
        "{} model accuracy (in %): {}".format(
            name, round(accuracy_score(y_test, y_pred) * 100, 3)
                   ),
        attrs=ATTRS,
        color=COLOR,
        )
    )
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        columns=["Predicted Healthy", "Predicted Parkinson's"],
        index=["True Healthy", "True Parkinson's"],
    )
    my_dict[name] = cm
print(cl("----------------------------------------", attrs=ATTRS, color=COLOR))
c=1
plt.figure(figsize=(10, 5))
for i in my_dict:
    plt.subplot(2, 3, c)
    sns.heatmap(
        my_dict[i],
        cmap="Spectral",
        linewidth=0.2,
        annot=True,
        cbar=False,
    )
    plt.title("Matrix for {}".format(i))
    c = c + 1
plt.tight_layout()
plt.show()