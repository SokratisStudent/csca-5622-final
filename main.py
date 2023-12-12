from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt
import numpy as np

# fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes)
feature_dataset = predict_students_dropout_and_academic_success.data.features
target_dataset = predict_students_dropout_and_academic_success.data.targets

combined_dataset = feature_dataset.copy()
target_column = []
for target_value in target_dataset['Target']:
    if target_value == 'Graduate':
        target_column.append(2)
    elif target_value == 'Enrolled':
        target_column.append(1)
    elif target_value == 'Dropout':
        target_column.append(0)

combined_dataset['Target'] = target_column
cm = combined_dataset.corr()

print(cm)

(train_features, test_features, train_targets, test_targets) = train_test_split(feature_dataset, target_dataset, test_size=0.2, random_state=80)


def build_dt(features, labels, max_depth=None, max_leaf_nodes=None, ccp_alpha=0.0):
    return DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, ccp_alpha=ccp_alpha).fit(features, labels)


def evaluate(dt, features, labels):
    y_pred = dt.predict(features)
    precision = precision_score(labels, y_pred, average='macro', zero_division=0.0)
    recall = recall_score(labels, y_pred, average='macro', zero_division=0.0)
    depth = dt.get_depth()
    l_nodes = dt.get_n_leaves()
    return (precision, recall, depth, l_nodes)


def evaluate_score(clf, features, labels):
    return clf.score(features, labels)


# Build the default tree and evaluate it as a baseline
dt = build_dt(train_features, train_targets)
(def_precision, def_recall, actual_depth, leaf_nodes) = evaluate(dt, test_features, test_targets)
def_score = evaluate_score(dt, test_features, test_targets)
print(f'Starting point - Decision Tree with {actual_depth=} and {leaf_nodes=} has precision={def_precision:f} and recall={def_recall:f} and {def_score=:f}')


current_best_precision = def_precision
current_best_recall = def_recall
current_best_score = def_score
current_best_max_depth = 0
max_max_depth = 100

for depth in range(1, max_max_depth):
    dt = build_dt(train_features, train_targets, max_depth=depth)
    score = evaluate_score(dt, test_features, test_targets)
    if score > current_best_score:
        current_best_score = score
        current_best_max_depth = depth

if current_best_max_depth == 0:
    current_best_max_depth = None

dt = build_dt(train_features, train_targets, max_depth=current_best_max_depth)
(current_best_precision, current_best_recall, actual_depth, leaf_nodes) = evaluate(dt, test_features, test_targets)
best_score = evaluate_score(dt, test_features, test_targets)
print(f'Max-depth tree - Decision Tree with {actual_depth=} and {leaf_nodes=} has precision={current_best_precision:f} and recall={current_best_recall:f} and {best_score=:f}')

current_best_max_leaf_nodes = 0
max_max_leaf = 500

for leaf in range(2, max_max_leaf):
    dt = build_dt(train_features, train_targets, max_depth=current_best_max_depth, max_leaf_nodes=leaf)
    score = evaluate_score(dt, test_features, test_targets)
    if score > current_best_score:
        current_best_score = score
        current_best_max_leaf_nodes = leaf

if current_best_max_leaf_nodes == 0:
    current_best_max_leaf_nodes = None

dt = build_dt(train_features, train_targets, max_leaf_nodes=current_best_max_leaf_nodes)
(current_best_precision, current_best_recall, actual_depth, leaf_nodes) = evaluate(dt, test_features, test_targets)
best_score = evaluate_score(dt, test_features, test_targets)
print(f'Pre-pruning tree - Decision Tree with {actual_depth=} and {leaf_nodes=} has precision={current_best_precision:f} and recall={current_best_recall:f} and {best_score=:f}')

path = dt.cost_complexity_pruning_path(train_features, train_targets)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
current_best_ccp_alphas = 0.0

for ccp_alpha in ccp_alphas:
    clf = build_dt(train_features, train_targets, max_depth=current_best_max_depth, max_leaf_nodes=current_best_max_leaf_nodes, ccp_alpha=ccp_alpha)
    score = evaluate_score(dt, test_features, test_targets)
    if score > current_best_score:
        current_best_score = score
        current_best_ccp_alphas = ccp_alphas

dt = build_dt(train_features, train_targets, max_depth=current_best_max_depth, max_leaf_nodes=current_best_max_leaf_nodes, ccp_alpha=current_best_ccp_alphas)
(current_best_precision, current_best_recall, actual_depth, leaf_nodes) = evaluate(dt, test_features, test_targets)
best_score = evaluate_score(dt, test_features, test_targets)
print(f'Post-pruning tree - Decision Tree with {actual_depth=} and {leaf_nodes=} has precision={current_best_precision:f} and recall={current_best_recall:f} and {best_score=:f}')

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

print("\nDecision Tree\n")
test_results = dt.predict(test_features)
print(confusion_matrix(test_targets, test_results), "\n")

print('Overall accuracy:', accuracy_score(test_targets, test_results))
print('F1_score:', f1_score(test_targets, test_results, average='micro'))
print()
print('Dropout precision:', precision_score(test_targets, test_results, average='micro', labels=['Dropout']))
print('Enrolled precision:', precision_score(test_targets, test_results, average='micro', labels=['Enrolled']))
print('Graduate precision:', precision_score(test_targets, test_results, average='micro', labels=['Graduate']))
print()
print('Dropout recall:', recall_score(test_targets, test_results, average='micro', labels=['Dropout']))
print('Enrolled recall:', recall_score(test_targets, test_results, average='micro', labels=['Enrolled']))
print('Graduate recall:', recall_score(test_targets, test_results, average='micro', labels=['Graduate']))


est_search_range = np.arange(5, 55, step=5)
lr_search_range = np.linspace(0.1, 1, num=10)
parameters = {'n_estimators': est_search_range, 'learning_rate': lr_search_range}
print(parameters)


def createAdaBoost(features, labels, n_estimators=50, learning_rate=1.0):
    return AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate).fit(features, labels.values.ravel())


grid = GridSearchCV(AdaBoostClassifier(estimator=dt), param_grid=parameters, cv=3, verbose=1).fit(train_features, train_targets.values.ravel())
print(grid.best_score_, grid.best_params_)

ada_boost = createAdaBoost(train_features, train_targets, n_estimators=grid.best_params_['n_estimators'], learning_rate=grid.best_params_['learning_rate'])

print("\nAda-Boosted DT\n")
test_results = ada_boost.predict(test_features)
print(confusion_matrix(test_targets, test_results), "\n")

print('Overall accuracy:', accuracy_score(test_targets, test_results))
print('F1_score:', f1_score(test_targets, test_results, average='micro'))
print()
print('Dropout precision:', precision_score(test_targets, test_results, average='micro', labels=['Dropout']))
print('Enrolled precision:', precision_score(test_targets, test_results, average='micro', labels=['Enrolled']))
print('Graduate precision:', precision_score(test_targets, test_results, average='micro', labels=['Graduate']))
print()
print('Dropout recall:', recall_score(test_targets, test_results, average='micro', labels=['Dropout']))
print('Enrolled recall:', recall_score(test_targets, test_results, average='micro', labels=['Enrolled']))
print('Graduate recall:', recall_score(test_targets, test_results, average='micro', labels=['Graduate']))


# bootstrap is set to false because we have already saved 20% of the dataset for testing so I want the forests to be trained on the entire dataset.
def build_random_forest(features, labels, max_depth=None, max_leaf_nodes=None, ccp_alpha=0.0):
    return RandomForestClassifier(bootstrap=False, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, ccp_alpha=ccp_alpha).fit(features, labels.values.ravel())

rf = build_random_forest(train_features, train_targets)
def_score = evaluate_score(rf, test_features, test_targets)
print(f'Starting point - Random Forest with has {def_score=:f}')

# Running this takes ~60 minutes so I'm just showing the result here.
#search_range = [ i for i in range(2, 51) ]
#parameters = {'max_depth': search_range, 'max_leaf_nodes': search_range}
#grid = GridSearchCV(rf, param_grid=parameters, cv=3, verbose=1).fit(train_features, train_targets.values.ravel())
#print(grid.best_score_, grid.best_params_)
print("0.7668837423268786 {'max_depth': 41, 'max_leaf_nodes': 47}")

rf_grid = build_random_forest(train_features, train_targets, max_depth=41, max_leaf_nodes=47)
def_score = evaluate_score(rf_grid, test_features, test_targets)
print(f'After doing grid search - Random Forest with has {def_score=:f}')

print("\nRandom Forest\n")
test_results = rf.predict(test_features)
print(confusion_matrix(test_targets, test_results), "\n")

print('Overall accuracy:', accuracy_score(test_targets, test_results))
print('F1_score:', f1_score(test_targets, test_results, average='micro'))
print()
print('Dropout precision:', precision_score(test_targets, test_results, average='micro', labels=['Dropout']))
print('Enrolled precision:', precision_score(test_targets, test_results, average='micro', labels=['Enrolled']))
print('Graduate precision:', precision_score(test_targets, test_results, average='micro', labels=['Graduate']))
print()
print('Dropout recall:', recall_score(test_targets, test_results, average='micro', labels=['Dropout']))
print('Enrolled recall:', recall_score(test_targets, test_results, average='micro', labels=['Enrolled']))
print('Graduate recall:', recall_score(test_targets, test_results, average='micro', labels=['Graduate']))

exp = export_text(dt, feature_names=feature_dataset.columns)
with open('tree_export', 'w') as f:
    f.write(exp)
    f.close()

plot_tree(dt)
plt.savefig('decision_tree.png', bbox_inches='tight', dpi=300)
plt.show()

print(exp)