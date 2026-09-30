from freight_cost_prediction.data_preprocessing import (
    load_data_from_db,
    preprocess_data,
    split_data
)

df = load_data_from_db("data/inventory.db", "vendor_invoice")

print(df.head())
print(df.shape)

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

print(X_train.shape)
print(X_test.shape)