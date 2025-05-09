import warnings
import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import precision_score, silhouette_score

# Import des modèles Scikit-Learn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Import XGBoost & LightGBM
from xgboost import XGBClassifier, XGBRegressor
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None

# Désactivation des warnings et des logs d’info
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('xgboost').setLevel(logging.ERROR)

# Liste des modèles à tester
hyperparameter_functions = {
    'KMeans', 'DBSCAN', 'AgglomerativeClustering',
    'RandomForestClassifier', 'GradientBoostingClassifier', 'AdaBoostClassifier',
    'LogisticRegression', 'LinearRegression', 'Lasso', 'Ridge',
    'SVC', 'SVR', 'DecisionTreeClassifier', 'DecisionTreeRegressor',
    'MLPClassifier', 'MLPRegressor',
    'XGBClassifier', 'XGBRegressor'
}
if LGBMClassifier:
    hyperparameter_functions.update({'LGBMClassifier', 'LGBMRegressor'})

# Mapping nom → constructeur
model_constructors = {
    'KMeans': KMeans,
    'DBSCAN': DBSCAN,
    'AgglomerativeClustering': AgglomerativeClustering,
    'RandomForestClassifier': RandomForestClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'AdaBoostClassifier': AdaBoostClassifier,
    'LogisticRegression': LogisticRegression,
    'LinearRegression': LinearRegression,
    'Lasso': Lasso,
    'Ridge': Ridge,
    'SVC': SVC,
    'SVR': SVR,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'MLPClassifier': MLPClassifier,
    'MLPRegressor': MLPRegressor,
    'XGBClassifier': XGBClassifier,
    'XGBRegressor': XGBRegressor,
}
if LGBMClassifier:
    model_constructors.update({
        'LGBMClassifier': LGBMClassifier,
        'LGBMRegressor': LGBMRegressor,
    })

# Grilles d’hyperparamètres
param_grids = {
    'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'random_state': [0]},
    'LogisticRegression': {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear'], 'random_state': [0]},
    'RandomForestClassifier': {'n_estimators': [50, 100], 'max_depth': [None, 10], 'random_state': [0]},
    'GradientBoostingClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]},
    'AdaBoostClassifier': {'n_estimators': [50, 100], 'learning_rate': [1.0, 0.1]},
    'DecisionTreeClassifier': {'max_depth': [None, 5, 10], 'random_state': [0]},
    'MLPClassifier': {'hidden_layer_sizes': [(50,), (100,)], 'max_iter': [200]},
    'KMeans': {'n_clusters': [2, 3, 4], 'random_state': [0]},
    'DBSCAN': {'eps': [0.5, 1.0], 'min_samples': [5, 10]},
    'AgglomerativeClustering': {'n_clusters': [2, 3, 4], 'linkage': ['ward', 'complete']},
    'LinearRegression': {},
    'Lasso': {'alpha': [0.1, 1.0]},
    'Ridge': {'alpha': [0.1, 1.0]},
    'SVR': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'DecisionTreeRegressor': {'max_depth': [None, 5, 10]},
    'MLPRegressor': {'hidden_layer_sizes': [(50,), (100,)], 'max_iter': [200]},
    'XGBClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]},
    'XGBRegressor': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]},
}
if LGBMClassifier:
    param_grids.update({
        'LGBMClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]},
        'LGBMRegressor': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]},
    })

# Chargement des datasets
data_clf = load_breast_cancer()
Xc, Xc_test, yc, yc_test = train_test_split(data_clf.data, data_clf.target,
                                            test_size=0.3, random_state=0)
data_reg = load_diabetes()
Xr, Xr_test, yr, yr_test = train_test_split(data_reg.data, data_reg.target,
                                            test_size=0.3, random_state=0)

# Collecte des résultats
results = []
for model_name in hyperparameter_functions:
    ModelClass = model_constructors[model_name]

    # Choix du jeu de données selon le modèle
    if model_name.endswith('Classifier') or model_name in ('SVC', 'LogisticRegression'):
        X_train, X_test, y_train, y_test, task = Xc, Xc_test, yc, yc_test, 'classification'
    elif model_name.endswith('Regressor') or model_name in ('LinearRegression','SVR','Lasso','Ridge'):
        X_train, X_test, y_train, y_test, task = Xr, Xr_test, yr, yr_test, 'regression'
    else:
        X_train, X_test, y_train, y_test, task = Xc, Xc_test, yc, yc_test, 'clustering'

    # 1) Sans hyperparam explicites
    model = ModelClass()
    t0 = time.perf_counter()
    model.fit(X_train, y_train) if task!='clustering' else model.fit(X_train)
    elapsed = time.perf_counter() - t0

    if task == 'classification':
        prec = precision_score(y_test, model.predict(X_test), average='macro')
    elif task == 'regression':
        prec = float('nan')
    else:
        labels = model.labels_
        prec = silhouette_score(X_train, labels) if len(set(labels))>1 else float('nan')

    results.append({
        'model': model_name,
        'explicit': False,
        'params': {},
        'fit_time_s': elapsed,
        'precision_macro': prec,
        'task': task
    })

    # 2) Avec hyperparam explicites
    for params in ParameterGrid(param_grids.get(model_name, {})):
        model = ModelClass(**params)
        t0 = time.perf_counter()
        model.fit(X_train, y_train) if task!='clustering' else model.fit(X_train)
        elapsed = time.perf_counter() - t0

        if task == 'classification':
            prec = precision_score(y_test, model.predict(X_test), average='macro')
        elif task == 'regression':
            prec = float('nan')
        else:
            labels = model.labels_
            prec = silhouette_score(X_train, labels) if len(set(labels))>1 else float('nan')

        results.append({
            'model': model_name,
            'explicit': True,
            'params': params,
            'fit_time_s': elapsed,
            'precision_macro': prec,
            'task': task
        })

# Création du DataFrame
df = pd.DataFrame(results)
#print(df.to_string(index=False))

# --- Synthèse pour tous les modèles (corrigée) ---
summary_rows = []
for model in df['model'].unique():
    sub = df[df['model'] == model]
    # configuration par défaut
    default = sub[sub['explicit'] == False].iloc[0]

    # configurations explicites valides (précision non-NaN)
    explicits = sub[sub['explicit'] == True]
    valid = explicits[~explicits['precision_macro'].isna()]

    if not valid.empty:
        # on prend la meilleure précision parmi les explicites valides
        best = valid.loc[valid['precision_macro'].idxmax()]
    else:
        # sinon on retombe sur la configuration par défaut
        best = default

    summary_rows.append({
        'Model': model,
        'Default Precision': default['precision_macro'],
        'Default Time (s)': default['fit_time_s'],
        'Best Precision': best['precision_macro'],
        'Best Time (s)': best['fit_time_s'],
        'Best Params': best['params']
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
