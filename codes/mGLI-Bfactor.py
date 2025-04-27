import numpy as np
from GLI_Functions import (
    get_protein_ca_atom_coordinate,
    GLI_feature,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import pandas as pd
import argparse
import os

import warnings

warnings.filterwarnings("ignore")


def normalize_feature(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)


def fitting(r, X_norm, y, model):
    regressor = model.fit(X_norm, y)
    y_pred = regressor.predict(X_norm)
    return pearsonr(y_pred, y)[0]


def b_factor_analysis(pdbid, datapath):
    filepath = datapath + "/" + pdbid + "_CA_A2.pdb"
    CA_coor, labels = get_protein_ca_atom_coordinate(pdbid, filepath)
    # print(CA_coor.shape[0])
    r_range = range(5, 27)
    GLIs = []
    for r in r_range:
        GLI = np.array(GLI_feature(CA_coor, r))
        GLIs.append(GLI)

    X = np.array(GLIs).T
    y = np.array(labels)

    X_norm = normalize_feature(X)
    print("mGLI feature generated for ", pdbid)

    model = LinearRegression()
    r = 27
    pvalue = fitting(r, X_norm, y, model)
    return pvalue


def cli_main():
    parser = argparse.ArgumentParser(description="Get KDA features for pdbbind")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Set-364",
        help="choose one of the three datasets including Bfactor-Set364, Bfactor-large, Bfactor-medium, Bfactor-small",
    )

    args = parser.parse_args()

    dataset_path = f"./datasets/{args.dataset_name}"
    df = pd.read_csv(f"{dataset_path}/{args.dataset_name}.csv", header=0)
    PDBIDs = df["PDBID"]
    pvalues = []
    for pdbid in PDBIDs:
        print(pdbid)
        pvalue = b_factor_analysis(pdbid, f"{dataset_path}/PDBs")
        pvalues.append(pvalue)

    if not os.path.exists("results-Bfactor"):
        os.mkdir("results-Bfactor")
    fw = open(f"results-Bfactor/results-{args.dataset_name}.csv", "w")
    print("PDB,PCC", file=fw)
    for pdbid, pvalues in zip(PDBIDs, pvalues):
        print(f"{pdbid},{pvalues:.3f}", file=fw)
    fw.close()


if __name__ == "__main__":
    cli_main()
    print("End!")
