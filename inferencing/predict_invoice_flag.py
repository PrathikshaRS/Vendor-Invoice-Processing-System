import joblib
import pandas as pd

MODEL_PATH = "models/random_forest_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice flagging model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model


def predict_invoice_flag(input_data):
    """
    Predict whether an invoice should be flagged.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted invoice flag
    """

    model = load_model()

    input_df = pd.DataFrame(input_data)

    input_df["Predicted_Flag"] = model.predict(input_df)

    return input_df


if __name__ == "__main__":

    # Example
    sample_data = {
        "invoice_quantity": [500, 1000],
        "invoice_dollars": [18500, 9000],
        "Freight": [100, 50],
        "days_po_to_invoice": [15, 12],
        "total_item_quantity": [500, 1000],
        "total_item_dollars": [18420, 9000],
        "avg_receiving_delay": [12, 7]
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)