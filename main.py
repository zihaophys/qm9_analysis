import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error

from qm9_properties import QM9_Properties, compounds

REP = 'CM' # 'CM' or 'BOB'
DATA_SIZE = 1000
TEST_SIZE = 0.2
VAL_SIZE = 0.2
seed = 1

# Data Preprocessing 
# For every compound generate a CM/BOB
if REP == 'CM':
    for mol in compounds[:DATA_SIZE]:
        mol.generate_coulomb_matrix(size=30, sorting="row-norm")
elif REP == 'BOB':
    for mol in compounds[:DATA_SIZE]:
        mol.generate_bob(size=30, asize={"O":3, "C":7, "N":3, "H":16, "S":1})
    
# stupid adjustment for data type conversion.
X = np.array([mol.representation for mol in compounds])[:DATA_SIZE]
XX = np.zeros((DATA_SIZE, X[0].shape[0]))
for i in range(DATA_SIZE):
    XX[i, :] = X[i]
X = XX


property_U = np.array([float(QM9_Properties[i - 1]['U']) for i in range(1, len(QM9_Properties) + 1)])[:DATA_SIZE]
property_U0 = np.array([float(QM9_Properties[i - 1]['U0']) for i in range(1, len(QM9_Properties) + 1)])[:DATA_SIZE]
property_alpha = np.array([float(QM9_Properties[i - 1]['alpha']) for i in range(1, len(QM9_Properties) + 1)])[:DATA_SIZE]

properties = [property_U, property_U0, property_alpha]
names = ['U', 'U0', 'alpha']

# Machine Learning 

print("======Kernel Ridge Regression======")
width_list = np.logspace(-3, 12, 20, base=2)

for name in names:

    print(name)

    X_train, X_test, Y_train, Y_test = train_test_split(X, properties[names.index(name)], test_size=TEST_SIZE, random_state=seed)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=VAL_SIZE, random_state=seed)

    mae_list = np.zeros(len(width_list))
    print("===Performing Validation===")
    for width in width_list:
        regr = KernelRidge(alpha=1e-7, kernel='rbf', gamma=1/width)
        regr.fit(X_train, Y_train)
        rg_MAE = mean_absolute_error(Y_val, regr.predict(X_val))
        print(" width: ", width, " MAE: ", rg_MAE)
        mae_list[width_list.tolist().index(width)] = rg_MAE
    min_mae = np.min(mae_list)
    location = np.where(mae_list == min_mae)
    opt_width = width_list[location[0][0]]
    print("After validation")
    print("Minimum MAE: ", min_mae, " at width: ", width_list[location[0][0]])

    print("===Training...===")
    X_train, X_test, Y_train, Y_test = train_test_split(X, properties[names.index(name)], test_size=TEST_SIZE, random_state=seed)

    regr = KernelRidge(alpha=1e-7, kernel='rbf', gamma=1/width)
    regr.fit(X_train, Y_train)
    rg_MAE = mean_absolute_error(Y_test, regr.predict(X_test))
    print("MAE: ", rg_MAE)

    print("==================")

print("=======88========")

print("======Random Forest======")

n_estimators_list = np.linspace(20, 150, 10, dtype=int)
min_samples_leaf_list = np.linspace(2, 15, 13, dtype=int)

for name in names:

    print(name)
    X_train, X_test, Y_train, Y_test = train_test_split(X, properties[names.index(name)], test_size=TEST_SIZE, random_state=seed)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=VAL_SIZE, random_state=1)
    mae_matrix = np.zeros((len(n_estimators_list), len(min_samples_leaf_list)))
    print("===Performing Validation===")
    for n_estimators in n_estimators_list:
        for min_samples_leaf in min_samples_leaf_list:
            regr = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
            regr.fit(X_train, Y_train)
            rf_MAE = mean_absolute_error(Y_val, regr.predict(X_val))
            print(" n_trees: ", n_estimators, " min_samples_leaf: ", min_samples_leaf, " MAE: ", rf_MAE)
            mae_matrix[n_estimators_list.tolist().index(n_estimators), min_samples_leaf_list.tolist().index(
                min_samples_leaf)] = rf_MAE

    min_mae = np.min(mae_matrix)
    location = np.where(mae_matrix == min_mae)
    opt_n_estimators = n_estimators_list[location[0][0]]
    opt_min_samples_leaf = min_samples_leaf_list[location[1][0]]
    print("After validation")
    print("Minimum MAE: ", min_mae, " at n_estimators: ", n_estimators_list[location[0][0]], " min_samples_leaf: ",
          min_samples_leaf_list[location[1][0]])

    print("===Training...===")
    regr = RandomForestRegressor(n_estimators=opt_n_estimators, min_samples_leaf=opt_min_samples_leaf)
    X_train, X_test, Y_train, Y_test = train_test_split(X, properties[names.index(name)], test_size=TEST_SIZE, random_state=seed)
    regr.fit(X_train, Y_train)
    rf_MAE = mean_absolute_error(Y_test, regr.predict(X_test))
    print("MAE: ", rf_MAE)

    print("==================")
print("=======88========")

print()