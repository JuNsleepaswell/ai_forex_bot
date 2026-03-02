# src/06_hyperparameter_tuning.py (Corrected and More Robust Version)

import numpy as np
import keras
from keras import layers
import keras_tuner as kt
import argparse
import os


def build_model(hp):
    """Builds a GRU model with tunable hyperparameters and gradient clipping."""

    hp_units_1 = hp.Int('units_1', min_value=32, max_value=96, step=16)
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=96, step=16)
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    model = keras.Sequential([
        keras.Input(shape=(240, 17)),
        layers.GRU(units=hp_units_1, return_sequences=True),
        layers.Dropout(rate=hp_dropout_1),
        layers.GRU(units=hp_units_2, return_sequences=False),
        layers.Dropout(rate=hp_dropout_2),
        layers.Dense(20, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    # --- THE FIX IS HERE: ADD GRADIENT CLIPPING ---
    # clipnorm=1.0 prevents the gradients from exploding, which is the likely
    # cause of all trials failing with NaN loss.
    optimizer = keras.optimizers.Adam(
        learning_rate=hp_learning_rate,
        clipnorm=1.0
    )

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def main(ticker):
    print(f"\n--- Step 6: Hyperparameter Tuning for {ticker} ---")

    PROCESSED_DATA_PATH = f'data/{ticker}_processed_data.npz'

    try:
        with np.load(PROCESSED_DATA_PATH, allow_pickle=True) as data:
            X_train = data['X_train']
            y_train = data['y_train']
            X_val = data['X_val']
            y_val = data['y_val']
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {PROCESSED_DATA_PATH}")
        return

    tuner = kt.RandomSearch(

        hypermodel=build_model,
        objective='val_loss',
        max_trials=20,
        executions_per_trial=1,
        directory='hyper_tuning',
        project_name=f'{ticker}_tuning',
        overwrite=False  # Set to True if you want to force a re-run
    )

    tuner.search_space_summary()

    print(f"\nStarting hyperparameter search for {ticker}...")
    tuner.search(
        X_train, y_train,
        batch_size=128,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
    )

    print(f"\n--- Tuning Complete for {ticker} ---")
    tuner.results_summary()

    # --- THE SECOND FIX IS HERE: ADD A ROBUSTNESS CHECK ---
    # This checks if the tuner actually found any valid models before trying to access them.
    best_hps_list = tuner.get_best_hyperparameters(1)
    if not best_hps_list:
        print("\nERROR: Could not find any successful models. All trials may have failed.")
        print("This is often caused by exploding gradients (NaN loss).")
        print("Try adjusting the search space (e.g., lower learning rates) or checking the data.")
    else:
        best_hps = best_hps_list[0]
        print("\n--- Best Hyperparameters Found ---")
        print(f"GRU Units 1:     {best_hps.get('units_1')}")
        print(f"Dropout Rate 1:  {best_hps.get('dropout_1'):.2f}")
        print(f"GRU Units 2:     {best_hps.get('units_2')}")
        print(f"Dropout Rate 2:  {best_hps.get('dropout_2'):.2f}")
        print(f"Learning Rate:   {best_hps.get('learning_rate')}")
        print("------------------------------------")
        print("\nNext step: Update your training script with these values and retrain the final model.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform hyperparameter tuning for a GRU model.')
    parser.add_argument('--ticker', type=str, required=True, help='The ticker symbol to tune.')
    args = parser.parse_args()
    main(args.ticker)