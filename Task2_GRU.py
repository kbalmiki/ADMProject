import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

# ===================== LOAD same task1 processed =====================
train_df = pd.read_csv("/ADMProject/processed_train.csv")
test_df  = pd.read_csv("/ADMProject/processed_test.csv")
train_df = train_df.dropna(subset=["label_id_final"]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["label_id_final"]).reset_index(drop=True)
# split features
feat_cols = [c for c in train_df.columns if c.startswith("f")]
X_train = train_df[feat_cols].values
X_test  = test_df[feat_cols].values

# label
y_train = train_df["label_id_final"].values - 1   # make 0 indexed
y_test  = test_df["label_id_final"].values - 1

num_classes = len(np.unique(y_train))

# reshape: (samples, timesteps, features_per_step)
# 240 -> 60 timesteps x 4 dims
X_train = X_train.reshape(-1, 60, 4)
X_test  = X_test.reshape(-1, 60, 4)

y_train_oh = to_categorical(y_train, num_classes)
y_test_oh  = to_categorical(y_test, num_classes)

# ===================== GRU MODEL =====================
model = Sequential([
    GRU(128, return_sequences=False, input_shape=(60,4)),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train_oh, epochs=30, batch_size=32, validation_split=0.2)

# ===================== Evaluation =====================
pred = np.argmax(model.predict(X_test), axis=1)
print("Test Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
