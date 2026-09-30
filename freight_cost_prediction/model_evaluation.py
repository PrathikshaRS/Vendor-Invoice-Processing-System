from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, max_depth=5):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, max_depth=5):
    model = RandomForestRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name: str)-> dict:
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)*100
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5

    print(f"\n{model_name} Performance:")
    print(f"R² Score: {r2:.2f}%")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    return {
        "model_name": model_name,
        "r2_score": r2,
        "mean_absolute_error": mae,
        "root_mean_squared_error": rmse
    }
