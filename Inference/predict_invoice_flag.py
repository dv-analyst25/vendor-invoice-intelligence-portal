import joblib
import pandas as pd

MODEL_PATH = ".//models//predict_flag_invoice.pkl"
SCALER_PATH = ".//models//scaler.pkl"

def load_model(model_path=MODEL_PATH):
    return joblib.load(model_path)


def load_scaler(scaler_path=SCALER_PATH):
    return joblib.load(scaler_path)

def predict_invoice_flag(input_data):

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    input_scaled = scaler.transform(input_df)

    input_df["Predicted_Flag"] = model.predict(input_scaled)

    return input_df


if __name__ == "__main__":

    input_data = {
        "invoice_quantity": [100, 50],
        "invoice_dollars": [18500, 9000],
        "Freight": [500, 300],
        "total_item_quantity": [100, 50],
        "total_item_dollars": [18000, 9000]
    }

    prediction = predict_invoice_flag(input_data)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print(prediction.to_string(index=False))    