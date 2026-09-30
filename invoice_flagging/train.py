from data_preprocessing import (
    load_invoice_data,
    apply_label,
    split_data,
    scale_features
)

from evaluate_classification_model import (
    train_random_forest,
    evaluate_classifier
)

import joblib


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "days_po_to_invoice",
    "total_item_quantity",
    "total_item_dollars",
    "avg_receiving_delay"
]

TARGET = "flag_invoice"


def main():

    df = load_invoice_data()

    df = apply_label(df)

    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET
    )

    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        "models/scaler.pkl"
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"]
    }

    grid_search = train_random_forest(
        X_train_scaled,
        y_train,
        param_grid
    )

    evaluate_classifier(
        grid_search,
        X_test_scaled,
        y_test,
        "Random Forest Classifier with Grid Search"
    )

    joblib.dump(
        grid_search.best_estimator_,
        "models/random_forest_model.pkl"
    )


if __name__ == "__main__":
    main()