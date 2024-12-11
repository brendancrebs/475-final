SEED = 42
EPOCHS = 20

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train = pd.read_csv('./train.csv', header = 0, index_col = 0)
test = pd.read_csv('./test.csv', header = 0, index_col = 0)

num_rows_train = train.shape[0]
print(f"Number of train rows: {num_rows_train}")

num_rows_test = test.shape[0]
print(f"Number of test rows: {num_rows_test}")

train_features = train.iloc[:,2:74]
targets = train['bg+1:00'].values
test_features = test.iloc[:,2:74]
imputer = SimpleImputer(strategy='mean')
LSTM_train = imputer.fit_transform(train_features)
LSTM_test = imputer.transform(test_features.values)

original_data = LSTM_train
time_steps = 72
features = 1

samples = train.shape[0] 
reshaped_data = original_data.reshape(samples, time_steps, -1)

scaler = StandardScaler()
reshaped_data = scaler.fit_transform(reshaped_data.reshape(-1, reshaped_data.shape[-1])).reshape(reshaped_data.shape)

checkpoint_lstm = ModelCheckpoint('best_lstm.keras', save_best_only=True, monitor='val_loss')
early_stopping_lstm = EarlyStopping(monitor='val_loss', restore_best_weights=False, patience=3)

x_train_lstm, x_test_lstm, y_train_lstm, y_test_lstm = train_test_split(reshaped_data, targets, test_size=0.2, random_state=42)

LSTM_model = Sequential()
LSTM_model.add(LSTM(5, activation='relu', input_shape=(time_steps, reshaped_data.shape[2]))) 
LSTM_model.add(Dropout(0.2))
LSTM_model.add(Dense(1))

optimizer = Adam(learning_rate=0.001)
LSTM_model.compile(optimizer=optimizer, loss='mean_squared_error')

history_LSTM = LSTM_model.fit(x_train_lstm, y_train_lstm, epochs=EPOCHS, batch_size=32, validation_split=0.2,
          shuffle=True,
          callbacks=[checkpoint_lstm, early_stopping_lstm]
         )

x_train_lstm.shape, x_test_lstm.shape, y_train_lstm.shape, y_test_lstm.shape

LSTM_model.load_weights('./best_lstm.keras')

y_pred = LSTM_model.predict(x_test_lstm)
rmse = np.sqrt(mean_squared_error(y_test_lstm, y_pred))
y_pred_lstm = LSTM_model.predict(x_test_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))

ceckpoint_mlp = ModelCheckpoint('best_mlp.keras', save_best_only=True, monitor='val_loss')
early_stopping_mlp = EarlyStopping(monitor='val_loss', restore_best_weights=False, patience=3)

scaler = StandardScaler()
MLP_data = scaler.fit_transform(LSTM_train)
MLP_targets = targets 
x_train_mlp, x_test_mlp, y_train_mlp, y_test_mlp = train_test_split(MLP_data, MLP_targets, test_size=0.2, random_state=SEED)

MLP_model = Sequential()
MLP_model.add(Dense(16, activation='relu', input_shape=(x_train_mlp.shape[1],)))  
MLP_model.add(Dense(4, activation='relu'))  
MLP_model.add(Dense(1))  

optimizer = Adam(learning_rate=0.001)
MLP_model.compile(optimizer=optimizer, loss='mean_squared_error')

history_MLP = MLP_model.fit(x_train_mlp, y_train_mlp, epochs=EPOCHS, batch_size=32, validation_split=0.2,
          shuffle=True,
          callbacks=[ceckpoint_mlp, early_stopping_mlp]
         )

MLP_model.load_weights('./best_mlp.keras')

y_pred = MLP_model.predict(x_test_mlp)
rmse = np.sqrt(mean_squared_error(y_test_mlp, y_pred))
y_pred_mlp = MLP_model.predict(x_test_mlp)
rmse_mlp = np.sqrt(mean_squared_error(y_test_mlp, y_pred_mlp))

x_train_rf, y_train_rf = x_train_mlp, y_train_mlp
x_test_rf, y_test_rf = x_test_mlp, y_test_mlp
RF_model = RandomForestRegressor(n_estimators=100, random_state=SEED)
history_RF = RF_model.fit(x_train_rf, y_train_rf)

y_pred_rf = RF_model.predict(x_test_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))

print("Performance Comparison:")
print(f"LSTM RMSE: {rmse_lstm:.4f}")
print(f"MLP RMSE:  {rmse_mlp:.4f}")
print(f"RF RMSE:   {rmse_rf:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_LSTM.history['loss'], label='Train Loss (LSTM)')
axes[0].plot(history_LSTM.history['val_loss'], label='Val Loss (LSTM)')
axes[0].set_title("LSTM Training & Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history_MLP.history['loss'], label='Train Loss (MLP)')
axes[1].plot(history_MLP.history['val_loss'], label='Val Loss (MLP)')
axes[1].set_title("MLP Training & Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(y_test_lstm, y_pred_lstm, alpha=0.7, edgecolors='k')
axes[0].plot([y_test_lstm.min(), y_test_lstm.max()], [y_test_lstm.min(), y_test_lstm.max()], 'r--')
axes[0].set_title("LSTM: Actual vs Predicted")
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")

axes[1].scatter(y_test_mlp, y_pred_mlp, alpha=0.7, edgecolors='k')
axes[1].plot([y_test_mlp.min(), y_test_mlp.max()], [y_test_mlp.min(), y_test_mlp.max()], 'r--')
axes[1].set_title("MLP: Actual vs Predicted")
axes[1].set_xlabel("Actual")
axes[1].set_ylabel("Predicted")

axes[2].scatter(y_test_rf, y_pred_rf, alpha=0.7, edgecolors='k')
axes[2].plot([y_test_rf.min(), y_test_rf.max()], [y_test_rf.min(), y_test_rf.max()], 'r--')
axes[2].set_title("RF: Actual vs Predicted")
axes[2].set_xlabel("Actual")
axes[2].set_ylabel("Predicted")

plt.tight_layout()
plt.show()

residuals_lstm = y_test_lstm - y_pred_lstm.reshape(-1)
residuals_mlp = y_test_mlp - y_pred_mlp.reshape(-1)
residuals_rf = y_test_rf - y_pred_rf.reshape(-1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(residuals_lstm, kde=True, ax=axes[0], color='blue')
axes[0].set_title("LSTM Residuals Distribution")
axes[0].set_xlabel("Residual")

sns.histplot(residuals_mlp, kde=True, ax=axes[1], color='green')
axes[1].set_title("MLP Residuals Distribution")
axes[1].set_xlabel("Residual")

sns.histplot(residuals_rf, kde=True, ax=axes[2], color='orange')
axes[2].set_title("RF Residuals Distribution")
axes[2].set_xlabel("Residual")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(y_pred_lstm, residuals_lstm, alpha=0.7, edgecolors='k')
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_title("LSTM Residuals vs Predictions")
axes[0].set_xlabel("Predictions")
axes[0].set_ylabel("Residuals")

axes[1].scatter(y_pred_mlp, residuals_mlp, alpha=0.7, edgecolors='k')
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_title("MLP Residuals vs Predictions")
axes[1].set_xlabel("Predictions")
axes[1].set_ylabel("Residuals")

axes[2].scatter(y_pred_rf, residuals_rf, alpha=0.7, edgecolors='k')
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_title("RF Residuals vs Predictions")
axes[2].set_xlabel("Predictions")
axes[2].set_ylabel("Residuals")

plt.tight_layout()
plt.show()

model_names = ['LSTM', 'MLP', 'RF']
rmses = [rmse_lstm, rmse_mlp, rmse_rf]

fig, ax = plt.subplots(figsize=(7,5))
sns.barplot(x=model_names, y=rmses, palette='viridis', ax=ax)
ax.set_title("RMSE Comparison of Models")
ax.set_ylabel("RMSE")
for i, v in enumerate(rmses):
    ax.text(i, v + 0.01, f"{v:.2f}", horizontalalignment='center', color='black', fontweight='bold')

plt.tight_layout()
plt.show()