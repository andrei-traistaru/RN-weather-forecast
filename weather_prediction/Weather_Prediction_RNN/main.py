import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Importuri din modulele noastre
from Weather_Prediction_RNN.config import config
from Weather_Prediction_RNN.src.preprocessing.data_prep import get_processed_data
from Weather_Prediction_RNN.src.neural_network.model_builder import build_lstm_model

def main():
    print("--- 1. Preprocesare Date ---")
    X_train, y_train, X_test, y_test, scaler = get_processed_data()
    print(f"Date Train: {X_train.shape}, Date Test: {X_test.shape}")

    print("\n--- 2. Construire Model LSTM ---")
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        learning_rate=config.LEARNING_RATE
    )
    model.summary()

    print("\n--- 3. Antrenare ---")
    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )

    print("\n--- 4. Evaluare și Predicție ---")
    predictions = model.predict(X_test)
    
    # Inversare scalare pentru interpretare
    # Trebuie să reconstruim forma originală (n_samples, n_features) pentru scaler
    n_features = len(config.FEATURES) + 1 # +1 pentru target
    
    dummy_pred = np.zeros((len(predictions), n_features))
    dummy_actual = np.zeros((len(y_test), n_features))
    
    dummy_pred[:, 0] = predictions.flatten()
    dummy_actual[:, 0] = y_test
    
    pred_real = scaler.inverse_transform(dummy_pred)[:, 0]
    actual_real = scaler.inverse_transform(dummy_actual)[:, 0]

    # Calcul Metrici (Conform Slide 25)
    mae = mean_absolute_error(actual_real, pred_real)
    rmse = math.sqrt(mean_squared_error(actual_real, pred_real))
    
    print(f"\nREZULTATE FINALE:")
    print(f"MAE: {mae:.2f} °C")
    print(f"RMSE: {rmse:.2f} °C")

    # Salvare model
    model.save(config.MODEL_SAVE_PATH)
    print(f"Model salvat la: {config.MODEL_SAVE_PATH}")

    # Vizualizare
    plt.figure(figsize=(12, 6))
    plt.plot(actual_real[:100], label='Real', color='blue')
    plt.plot(pred_real[:100], label='Predicție', color='red', linestyle='--')
    plt.title('Predicție Meteo - Arhitectura Modulară')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()