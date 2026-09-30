from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score, accuracy_score


def train_random_forest(X_train, y_train, param_grid):
    """
    Train a Random Forest Classifier using GridSearchCV to find the best hyperparameters.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.
    param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.

    Returns:
    GridSearchCV: Fitted GridSearchCV object with the best model.
    """

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"]
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def evaluate_classifier(model, X_test, y_test, model_name):
    """
    Evaluate the classifier on the test set and print classification metrics.

    Parameters:
    model: Trained classifier.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target variable.
    model_name: Name of the model for reporting purposes.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)