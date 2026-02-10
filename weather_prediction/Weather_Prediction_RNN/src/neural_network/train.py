import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import joblib
from sklearn.metrics import r2_score

# --- 1. CONFIGURARE CAI RELATIVE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.data_prep import get_processed_data
from src.neural_network.model_builder import build_regression_model, get_callbacks
import config.config as config

def main():
    print("[INFO] Se porneste antrenarea sistemului meteo (Regresie)...")
    print(f"[INFO] Script locat in: {current_dir}")
    print(f"[INFO] Radacina proiect: {project_root}")
    
    # 1. Date
    try:
        data = get_processed_data()
        if len(data) == 8:
            X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y = data
        else:
            print("[EROARE] data_prep.py nu returneaza scaler_x.")
            return
    except Exception as e:
        print(f"[EROARE] Preprocesare: {e}")
        return

    # 2. Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_units = y_train.shape[-1] 
    
    print(f"[INFO] Construire model: input={input_shape}, output_vars={output_units}")
    
    model = build_regression_model(input_shape, output_units)
    callbacks = get_callbacks()
    
    # 3. Antrenare
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. Evaluare
    print("\n[INFO] Calculare metrici...")
    predictions_scaled = model.predict(X_test)
    
    N_test, steps, vars_count = predictions_scaled.shape
    
    pred_flat = predictions_scaled.reshape(-1, vars_count)
    y_test_flat = y_test.reshape(-1, vars_count)
    
    pred_real_flat = scaler_y.inverse_transform(pred_flat)
    y_test_real_flat = scaler_y.inverse_transform(y_test_flat)
    
    r2_temp = r2_score(y_test_real_flat[:, 0], pred_real_flat[:, 0])
    r2_hum = r2_score(y_test_real_flat[:, 1], pred_real_flat[:, 1])
    
    print(f"   R2 Score (Global pe 24h): Temp={r2_temp:.4f}, Hum={r2_hum:.4f}")
    
    # 5. Salvare Grafic
    os.makedirs(config.DOCS_PATH, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (Huber/MSE)')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(config.DOCS_PATH, 'loss_curve.png'))
    
    # 6. Salvare Model si Scalere
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    
    # (optimized_model)
    model.save(os.path.join(config.MODELS_PATH, 'optimized_model.keras'))
    joblib.dump(scaler_x, os.path.join(config.MODELS_PATH, 'scaler_x.save'))
    joblib.dump(scaler_y, os.path.join(config.MODELS_PATH, 'scaler_y.save'))
    
    print("\n[SUCCES] Antrenare completa. Modelul si scalerele au fost salvate!")

if __name__ == "__main__":
    main()