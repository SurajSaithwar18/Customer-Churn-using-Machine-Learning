import os
import sys
import matplotlib.pyplot as plt

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    make_scorer
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting Training and Testing Data"
            )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            print("\nTuning Logistic Regression...")

            param_grid = {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs"]
            }

            f1_scorer = make_scorer(
                f1_score,
                pos_label="Yes"
            )

            grid = GridSearchCV(
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42
                ),
                param_grid,
                cv=5,
                scoring=f1_scorer,
                n_jobs=-1
            )

            grid.fit(
                X_train,
                y_train
            )

            best_logistic = (
                grid.best_estimator_
            )

            print(
                "Best Logistic Parameters:",
                grid.best_params_
            )

            models = {

                "Logistic Regression":
                best_logistic,

                "Random Forest":
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    random_state=42
                ),

                "Gradient Boosting":
                GradientBoostingClassifier(
                    random_state=42
                ),

                "AdaBoost":
                AdaBoostClassifier(
                    random_state=42
                ),

                "SVM":
                SVC(
                    probability=True,
                    random_state=42
                )
            }

            best_model = None
            best_model_name = None
            best_f1 = 0
            best_roc_auc = 0

            for model_name, model in models.items():

                logging.info(
                    f"Training {model_name}"
                )

                model.fit(
                    X_train,
                    y_train
                )

                y_pred = model.predict(
                    X_test
                )

                y_prob = model.predict_proba(
                    X_test
                )[:, 1]

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                precision = precision_score(
                    y_test,
                    y_pred,
                    pos_label="Yes"
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    pos_label="Yes"
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    pos_label="Yes"
                )

                roc_auc = roc_auc_score(
                    (y_test == "Yes").astype(int),
                    y_prob
                )

                print("\n" + "=" * 50)
                print(model_name)
                print("=" * 50)

                print(f"Accuracy  : {accuracy:.4f}")
                print(f"Precision : {precision:.4f}")
                print(f"Recall    : {recall:.4f}")
                print(f"F1 Score  : {f1:.4f}")
                print(f"ROC AUC   : {roc_auc:.4f}")

                print("\nConfusion Matrix:")
                print(
                    confusion_matrix(
                        y_test,
                        y_pred
                    )
                )

                print("\nClassification Report:")
                print(
                    classification_report(
                        y_test,
                        y_pred
                    )
                )

                if f1 > best_f1:

                    best_f1 = f1
                    best_roc_auc = roc_auc

                    best_model = model
                    best_model_name = model_name

            print("\n" + "=" * 50)
            print(
                f"Best Model : {best_model_name}"
            )
            print(
                f"Best F1 Score : {best_f1:.4f}"
            )

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            RocCurveDisplay.from_estimator(
                best_model,
                X_test,
                y_test
            )

            plt.savefig(
                "artifacts/roc_curve.png"
            )

            plt.close()

            with open(
                "artifacts/metrics.txt",
                "w"
            ) as file:

                file.write(
                    f"Best Model: {best_model_name}\n"
                )

                file.write(
                    f"Best F1 Score: {best_f1:.4f}\n"
                )

                file.write(
                    f"ROC AUC: {best_roc_auc:.4f}\n"
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                "Best Model Saved Successfully"
            )

            return best_f1

        except Exception as e:
            raise CustomException(
                e,
                sys
            )