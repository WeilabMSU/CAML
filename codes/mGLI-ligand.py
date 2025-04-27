from Bio import PDB
import numpy as np
import pandas
import math
import sys
import time
import os
import argparse
import pickle

# elements in ligand to be considered
el_l = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"]
scales = [0, 2, 2.44, 2.98, 3.63, 4.43, 5.41, 6.59, 8.05, 10]


def Gauss_linking_integral(line1, line2):
    """
    a:tuple; elements are head and tail
    of line segment, each is a (3,) array representing the xyz coordinate.
    """
    # a0, a1 = a[0],a[1]
    # b0, b1 = b[0],b[1]
    a = [line1.startpoint, line1.endpoint]
    b = [line2.startpoint, line2.endpoint]

    R = np.empty((2, 2), dtype=tuple)
    for i in range(2):
        for j in range(2):
            R[i, j] = a[i] - b[j]

    n = []
    cprod = []

    cprod.append(np.cross(R[0, 0], R[0, 1]))
    cprod.append(np.cross(R[0, 1], R[1, 1]))
    cprod.append(np.cross(R[1, 1], R[1, 0]))
    cprod.append(np.cross(R[1, 0], R[0, 0]))

    for c in cprod:
        n.append(c / (np.linalg.norm(c) + 1e-6))

    area1 = np.arcsin(np.dot(n[0], n[1]))
    area2 = np.arcsin(np.dot(n[1], n[2]))
    area3 = np.arcsin(np.dot(n[2], n[3]))
    area4 = np.arcsin(np.dot(n[3], n[0]))

    sign = np.sign(np.cross(a[1] - a[0], b[1] - b[0]).dot(a[0] - b[0]))
    Area = area1 + area2 + area3 + area4

    return sign * Area


class Line:
    def __init__(self, startpoint, endpoint, start_type, end_type):
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.start_type = start_type
        self.end_type = end_type


class Atom:
    def __init__(
        self, data, etype, eid=None, serial_id=None
    ):  # data is (3,) array, etype is str
        self.data = data
        self.etype = etype  # atom element
        self.eid = eid  # atom full name
        self.serial_id = serial_id  # serialid


class bonded_Atom:
    def __init__(self, lines, atom):  # (init with a list of line object)
        self.lines = lines
        self.atom = atom

    def add_line(self, line):
        self.lines.append(line)


def get_ligand_struct_from_mol2(mol2file):
    atoms, bondlist = get_atom_and_bond_list_from_mol2(mol2file)
    print("# of atoms in the ligand", len(atoms))
    bonded_atoms = []
    for atom in atoms:
        bonded_atom = bonded_Atom([], atom)
        bonded_atoms.append(bonded_atom)

    for item in bondlist:
        index1, index2 = item[0], item[1]
        startpoint = bonded_atoms[index1 - 1].atom.data
        endpoint = bonded_atoms[index2 - 1].atom.data
        start_type = bonded_atoms[index1 - 1].atom.etype
        end_type = bonded_atoms[index2 - 1].atom.etype

        midpoint = (startpoint + endpoint) / 2
        line1 = Line(startpoint, midpoint, start_type, end_type)
        line2 = Line(endpoint, midpoint, end_type, start_type)
        bonded_atoms[index1 - 1].add_line(line1)
        bonded_atoms[index2 - 1].add_line(line2)
    return bonded_atoms


def get_atom_and_bond_list_from_mol2(mol2file):
    from biopandas.mol2 import PandasMol2

    print(mol2file)
    contents = open(mol2file).read().splitlines()
    for idx, l in enumerate(contents):
        # if "SMILES" in l and contents[idx - 1] == "@<TRIPOS>MOLECULE":
        #     natom, nbonds = contents[idx + 1].split()[:2]
        #     break
        if "@<TRIPOS>BOND" in l:
            bond_start = idx + 1
        if "@<TRIPOS>SUBSTRUCTURE" in l:
            bond_stop = idx - 1
    nbonds = bond_stop - bond_start + 1

    print(mol2file, nbonds)

    bondlist = []
    for idx, l in enumerate(contents):
        if l == "@<TRIPOS>BOND":
            for i in range(int(nbonds)):
                ll = contents[idx + i + 1]
                info_bond = np.array(ll.split()[1:3]).astype(int).tolist()
                bondlist.append(info_bond)

    df_atoms = PandasMol2().read_mol2(mol2file).df

    def element_map(e_old):
        e_new = e_old.split(".")[0]
        return e_new

    df_atoms["e_map"] = df_atoms["atom_type"].apply(element_map)

    atoms = []
    for pos, etype, serial_id in zip(
        df_atoms.loc[:, ["x", "y", "z"]].values, df_atoms["e_map"], df_atoms["atom_id"]
    ):
        atom = Atom(pos, etype, etype, serial_id)
        atoms.append(atom)

    return atoms, bondlist


class Gauss_linking_integral_of_structures:
    def __init__(self, struct1, struct2, integral_type):
        self.struct1 = struct1
        self.struct2 = struct2
        self.result = {}
        self.integral_type = integral_type
        self.calculate_result()

    def calculate_result(self):
        for bonded_atom1 in self.struct1:
            for bonded_atom2 in self.struct2:
                L = []
                for line1 in bonded_atom1.lines:
                    for line2 in bonded_atom2.lines:
                        gli = Gauss_linking_integral(line1, line2)
                        L.append(np.abs(gli))

                res = np.median(L)

                self.result[
                    bonded_atom1.atom.serial_id, bonded_atom2.atom.serial_id
                ] = res

    def get_result(self, bonded_atom1, bonded_atom2):
        return self.result[bonded_atom1.atom.serial_id, bonded_atom2.atom.serial_id]


def get_inteval(args, ep, el):

    intevals = []
    if ep in el_l and el in el_l:
        for i in range(len(scales) - 1):
            r1 = scales[i]
            r2 = scales[i + 1]
            intevals.append([r1, r2])

    return intevals


def get_kda_feature_l(e1, e2, r1, r2, protein_struct, ligand_struct, F, args):
    Gli_sum = []
    for bonded_atom2 in ligand_struct:
        if bonded_atom2.atom.etype == e2:
            gli_sum = 0
            for bonded_atom1 in protein_struct:
                if bonded_atom1.atom.etype == e1:
                    dist = math.dist(bonded_atom1.atom.data, bonded_atom2.atom.data)
                    if args.bin_or_all == "bin":
                        if dist >= r1 and dist < r2:
                            gli_sum += F.result[
                                bonded_atom1.atom.serial_id, bonded_atom2.atom.serial_id
                            ]
                    elif args.bin_or_all == "all":
                        if dist < r2:
                            gli_sum += F.result[
                                bonded_atom1.atom.serial_id, bonded_atom2.atom.serial_id
                            ]
            Gli_sum.append(gli_sum)

    if Gli_sum == []:

        kda_feature = [0, 0, 0, 0, 0]

    else:
        Gli_sums = np.array(Gli_sum)

        kda_feature = [
            np.sum(Gli_sums),
            np.min(Gli_sums),
            np.max(Gli_sums),
            np.mean(Gli_sums),
            np.median(Gli_sums),
        ]

    return kda_feature


def get_KDA_features(args, mol2_path):

    ligand_struct = get_ligand_struct_from_mol2(mol2_path)
    F = Gauss_linking_integral_of_structures(
        ligand_struct, ligand_struct, args.integral_type
    )

    kda_feature = []
    for e1 in el_l:
        for e2 in el_l:
            for scale in get_inteval(args, e1, e2):
                r1, r2 = scale
                kda_feature_l = get_kda_feature_l(
                    e1, e2, r1, r2, ligand_struct, ligand_struct, F, args
                )
                kda_feature += kda_feature_l

    colss = []

    len_fps_pl = 5 * (len(scales) - 1)
    for idx, e1 in enumerate(el_l):
        for e2 in el_l[idx:]:
            idx_e1 = el_l.index(e1)
            idx_e2 = el_l.index(e2)
            col_start = (idx_e1 * len(el_l) + idx_e2) * len_fps_pl
            coL_stop = (idx_e1 * len(el_l) + idx_e2 + 1) * len_fps_pl
            cols = np.arange(col_start, coL_stop).tolist()
            colss += cols

    return np.array(kda_feature)[colss]


def main():
    parser = argparse.ArgumentParser(description="Get KDA features for pdbbind")
    parser.add_argument("--mol2_path", type=str)
    parser.add_argument("--mol2_id", type=str)
    parser.add_argument("--bin_or_all", type=str, default="bin", help="bin or all")
    parser.add_argument("--integral_type", type=str, default="median")
    args = parser.parse_args()

    kda_feature = get_KDA_features(args, args.mol2_path)

    print(np.shape(kda_feature))
    np.save(
        f"{args.mol2_id}-ligand-{args.integral_type}-{args.bin_or_all}", kda_feature
    )


if __name__ == "__main__":
    main()
    print("End!")
