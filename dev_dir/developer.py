import socket
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import LogisticRegression
from sklearn.metrics import accuracy_score
from shutil import copyfile
import os
import shutil

def copy_generated_specs():
    src = os.path.join(os.getcwd(), 'generated_specs')
    dest = os.path.join(os.getcwd(), '..', 'server_dir', 'generated_specs')

    if os.path.exists(dest):
        shutil.rmtree(dest)

    try:
        shutil.copytree(src, dest)
        print("Folder successfully copied to server_dir.")
    except Exception as e:
        print(f"An error occurred: {e}")


# development machine
# Load the dataset
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model quantised to 8 bits
model = LogisticRegression(n_bits=8)
model.fit(X_train, y_train)
model.compile(X_train)

print("model trained and compiled")

model_dev = FHEModelDev("generated_specs", model)
model_dev.save()

copy_generated_specs()
