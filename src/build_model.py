import numpy as np
import argparse
import pandas as pd
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import os
from sklearn.preprocessing import StandardScaler
import argparse

parser = argparse.ArgumentParser(description="PSR GBDT models")
parser.add_argument("--icycle", type=int, default=0)
parser.add_argument("--interaction_types", type=str, default="P-L")
parser.add_argument("--comb_type", type=str, default="ES", help="ES or CS")
parser.add_argument("--dataset_name", type=str, default="v2016")
parser.add_argument("--RPS_option", type=str, default="facet")
parser.add_argument("--min_edge_length", type=int, default=1)
parser.add_argument("--max_edge_length", type=int, default=12)
parser.add_argument("--num_samples", type=int, default=23)
parser.add_argument("--max_dim", type=int, default=1)
args = parser.parse_args()

np.random.seed(args.icycle)

ml_method = "GradientBoostingRegressor"

main_dir = "."

tasks = ["refine", "core"]
interaction_types = args.interaction_types.split("_")


max_dim = args.max_dim
RPS_options = args.RPS_option.split("_")
curve_or_rates = ["curves", "rates"]
features_train_test = [[], []]
for itask, task in enumerate(tasks):
    for interaction_type in interaction_types:
        for curve_or_rate in curve_or_rates:
            for RPS_option in RPS_options:
                for dim in range(max_dim + 1):
                    feature_path = f"{main_dir}/PSR-features/PSR-{args.dataset_name}-{task}_{RPS_option}-{curve_or_rate}-dim{dim}_{args.comb_type}-{interaction_type}-r{args.min_edge_length}-r{args.max_edge_length}-ns{args.num_samples}.npy"
                features = np.load(feature_path)
                features_train_test[itask].append(features)

X_train = np.concatenate(features_train_test[0], axis=1)
X_test = np.concatenate(features_train_test[1], axis=1)

data_path = f"{main_dir}/datasets"
y_train = pd.read_csv(
    f"{data_path}/PDBbind-v2016-refine.csv", header=0, index_col=False
)["label"].values.ravel()
y_test = pd.read_csv(f"{data_path}/PDBbind-v2016-core.csv", header=0, index_col=False)[
    "label"
].values.ravel()


use_norm = True
if use_norm:
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

num_tree = 20000
j = 7
k = 5
m = 8
lr = 0.002
clf = globals()["%s" % ml_method](
    n_estimators=num_tree,
    max_depth=j,
    min_samples_split=k,
    learning_rate=lr,
    subsample=0.1 * m,
    max_features="sqrt",
)

clf.fit(X_train, y_train)
yP = clf.predict(X_test)
corr, _ = pearsonr(yP, y_test)
rmse = np.sqrt(mean_squared_error(yP, y_test))

results_path = f"{main_dir}/results/{args.dataset_name}"
if not os.path.exists(results_path):
    os.mkdir(results_path)

fw = open(
    f"{results_path}/prediction-PSR-{args.dataset_name}-comb_{args.comb_type}-act_type_{args.interaction_types}-c{args.icycle}_{args.RPS_option}-curves-concat-rates-max_dim{max_dim}-r{args.min_edge_length}-r{args.max_edge_length}-ns{args.num_samples}.csv",
    "w",
)
print("R=%.3f RMSE=%.3f" % (corr, rmse), file=fw)
fw.close()

fw = open(
    f"{results_path}/BA-PSR-{args.dataset_name}-comb_{args.comb_type}-act_type_{args.interaction_types}-c{args.icycle}_{args.RPS_option}-curves-concat-rates-max_dim{max_dim}-r{args.min_edge_length}-r{args.max_edge_length}-ns{args.num_samples}.csv",
    "w",
)
for ba in yP:
    print(ba, file=fw)
fw.close()
