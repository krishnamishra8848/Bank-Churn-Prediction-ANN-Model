import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# 1. Load the data
df = pd.read_csv('/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv')

# 2. Remove unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 3. Prepare features (X) and target (y)
X = df.drop('Exited', axis=1)  # Features
y = df['Exited']  # Target

# 4. One-Hot Encoding and Standardization
# Separate categorical and numerical features
categorical_features = ['Geography', 'Gender']
numerical_features = X.columns.difference(categorical_features)

# One-Hot Encode categorical variables and standardize numerical ones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply the transformations
X_prepared = preprocessor.fit_transform(X)

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)

# 6. Build the ANN Model
def build_model():
    model = Sequential()
    # Hidden layer with 32 nodes, ReLU activation, and He normal initialization
    model.add(Dense(32, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1],)))
    # Output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))
    return model

# 7. Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch > 100:
        lr = lr * 0.5
    return lr

# Compile the model with Adam optimizer
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# 8. Train the model with Early Stopping and Learning Rate Scheduler
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=200, 
                    batch_size=32, 
                    callbacks=[early_stopping, lr_scheduler],
                    verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

250/250 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - accuracy: 0.6544 - loss: 0.6382 - val_accuracy: 0.8145 - val_loss: 0.4386 - learning_rate: 0.0010
Epoch 2/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8163 - loss: 0.4284 - val_accuracy: 0.8225 - val_loss: 0.4096 - learning_rate: 0.0010
Epoch 3/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8239 - loss: 0.4156 - val_accuracy: 0.8300 - val_loss: 0.3878 - learning_rate: 0.0010
Epoch 4/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8306 - loss: 0.3967 - val_accuracy: 0.8410 - val_loss: 0.3725 - learning_rate: 0.0010
Epoch 5/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8416 - loss: 0.3763 - val_accuracy: 0.8480 - val_loss: 0.3623 - learning_rate: 0.0010
Epoch 6/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8604 - loss: 0.3512 - val_accuracy: 0.8535 - val_loss: 0.3563 - learning_rate: 0.0010
Epoch 7/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8490 - loss: 0.3657 - val_accuracy: 0.8580 - val_loss: 0.3504 - learning_rate: 0.0010
Epoch 8/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8561 - loss: 0.3491 - val_accuracy: 0.8565 - val_loss: 0.3478 - learning_rate: 0.0010
Epoch 9/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8576 - loss: 0.3455 - val_accuracy: 0.8555 - val_loss: 0.3475 - learning_rate: 0.0010
Epoch 10/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8557 - loss: 0.3488 - val_accuracy: 0.8585 - val_loss: 0.3468 - learning_rate: 0.0010
Epoch 11/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8552 - loss: 0.3472 - val_accuracy: 0.8605 - val_loss: 0.3457 - learning_rate: 0.0010
Epoch 12/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8554 - loss: 0.3493 - val_accuracy: 0.8585 - val_loss: 0.3448 - learning_rate: 0.0010
Epoch 13/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8589 - loss: 0.3413 - val_accuracy: 0.8590 - val_loss: 0.3438 - learning_rate: 0.0010
Epoch 14/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8686 - loss: 0.3222 - val_accuracy: 0.8575 - val_loss: 0.3445 - learning_rate: 0.0010
Epoch 15/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8606 - loss: 0.3318 - val_accuracy: 0.8610 - val_loss: 0.3432 - learning_rate: 0.0010
Epoch 16/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8641 - loss: 0.3334 - val_accuracy: 0.8610 - val_loss: 0.3437 - learning_rate: 0.0010
Epoch 17/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8535 - loss: 0.3484 - val_accuracy: 0.8590 - val_loss: 0.3433 - learning_rate: 0.0010
Epoch 18/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8584 - loss: 0.3360 - val_accuracy: 0.8610 - val_loss: 0.3423 - learning_rate: 0.0010
Epoch 19/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8679 - loss: 0.3327 - val_accuracy: 0.8595 - val_loss: 0.3408 - learning_rate: 0.0010
Epoch 20/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8648 - loss: 0.3293 - val_accuracy: 0.8605 - val_loss: 0.3396 - learning_rate: 0.0010
Epoch 21/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8637 - loss: 0.3359 - val_accuracy: 0.8590 - val_loss: 0.3417 - learning_rate: 0.0010
Epoch 22/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8660 - loss: 0.3198 - val_accuracy: 0.8595 - val_loss: 0.3419 - learning_rate: 0.0010
Epoch 23/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8681 - loss: 0.3243 - val_accuracy: 0.8595 - val_loss: 0.3401 - learning_rate: 0.0010
Epoch 24/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8557 - loss: 0.3427 - val_accuracy: 0.8610 - val_loss: 0.3384 - learning_rate: 0.0010
Epoch 25/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8574 - loss: 0.3365 - val_accuracy: 0.8630 - val_loss: 0.3382 - learning_rate: 0.0010
Epoch 26/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8612 - loss: 0.3285 - val_accuracy: 0.8605 - val_loss: 0.3386 - learning_rate: 0.0010
Epoch 27/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8641 - loss: 0.3256 - val_accuracy: 0.8615 - val_loss: 0.3382 - learning_rate: 0.0010
Epoch 28/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8633 - loss: 0.3331 - val_accuracy: 0.8620 - val_loss: 0.3371 - learning_rate: 0.0010
Epoch 29/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8693 - loss: 0.3224 - val_accuracy: 0.8615 - val_loss: 0.3390 - learning_rate: 0.0010
Epoch 30/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8674 - loss: 0.3194 - val_accuracy: 0.8595 - val_loss: 0.3375 - learning_rate: 0.0010
Epoch 31/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8646 - loss: 0.3273 - val_accuracy: 0.8640 - val_loss: 0.3378 - learning_rate: 0.0010
Epoch 32/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8667 - loss: 0.3335 - val_accuracy: 0.8630 - val_loss: 0.3383 - learning_rate: 0.0010
Epoch 33/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8709 - loss: 0.3145 - val_accuracy: 0.8645 - val_loss: 0.3404 - learning_rate: 0.0010
Epoch 34/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8673 - loss: 0.3285 - val_accuracy: 0.8625 - val_loss: 0.3369 - learning_rate: 0.0010
Epoch 35/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8712 - loss: 0.3203 - val_accuracy: 0.8620 - val_loss: 0.3373 - learning_rate: 0.0010
Epoch 36/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8666 - loss: 0.3181 - val_accuracy: 0.8640 - val_loss: 0.3394 - learning_rate: 0.0010
Epoch 37/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8635 - loss: 0.3331 - val_accuracy: 0.8645 - val_loss: 0.3372 - learning_rate: 0.0010
Epoch 38/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8651 - loss: 0.3265 - val_accuracy: 0.8605 - val_loss: 0.3392 - learning_rate: 0.0010
Epoch 39/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8558 - loss: 0.3389 - val_accuracy: 0.8590 - val_loss: 0.3380 - learning_rate: 0.0010
Epoch 40/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.8694 - loss: 0.3118 - val_accuracy: 0.8650 - val_loss: 0.3373 - learning_rate: 0.0010
Epoch 41/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8649 - loss: 0.3246 - val_accuracy: 0.8615 - val_loss: 0.3367 - learning_rate: 0.0010
Epoch 42/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8690 - loss: 0.3198 - val_accuracy: 0.8625 - val_loss: 0.3376 - learning_rate: 0.0010
Epoch 43/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8690 - loss: 0.3218 - val_accuracy: 0.8605 - val_loss: 0.3368 - learning_rate: 0.0010
Epoch 44/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8709 - loss: 0.3247 - val_accuracy: 0.8655 - val_loss: 0.3365 - learning_rate: 0.0010
Epoch 45/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8612 - loss: 0.3356 - val_accuracy: 0.8605 - val_loss: 0.3383 - learning_rate: 0.0010
Epoch 46/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8654 - loss: 0.3301 - val_accuracy: 0.8665 - val_loss: 0.3354 - learning_rate: 0.0010
Epoch 47/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8674 - loss: 0.3206 - val_accuracy: 0.8625 - val_loss: 0.3365 - learning_rate: 0.0010
Epoch 48/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8668 - loss: 0.3215 - val_accuracy: 0.8665 - val_loss: 0.3377 - learning_rate: 0.0010
Epoch 49/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8674 - loss: 0.3215 - val_accuracy: 0.8640 - val_loss: 0.3386 - learning_rate: 0.0010
Epoch 50/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8669 - loss: 0.3251 - val_accuracy: 0.8625 - val_loss: 0.3373 - learning_rate: 0.0010
Epoch 51/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8678 - loss: 0.3231 - val_accuracy: 0.8620 - val_loss: 0.3391 - learning_rate: 0.0010
Epoch 52/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8643 - loss: 0.3282 - val_accuracy: 0.8660 - val_loss: 0.3360 - learning_rate: 0.0010
Epoch 53/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8655 - loss: 0.3209 - val_accuracy: 0.8615 - val_loss: 0.3365 - learning_rate: 0.0010
Epoch 54/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8657 - loss: 0.3309 - val_accuracy: 0.8635 - val_loss: 0.3365 - learning_rate: 0.0010
Epoch 55/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8648 - loss: 0.3244 - val_accuracy: 0.8655 - val_loss: 0.3360 - learning_rate: 0.0010
Epoch 56/200
250/250 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8712 - loss: 0.3121 - val_accuracy: 0.8625 - val_loss: 0.3360 - learning_rate: 0.0010
63/63 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8660 - loss: 0.3379
Test Accuracy: 86.65%
