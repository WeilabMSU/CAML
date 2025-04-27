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
el_p = ["C", "N", "O", "S"]  # elements in protein to be considered


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


def get_bond(atoms, atom):
    maximal_bond_lengths = [
        ["C", "H", 1.19],
        ["C", "C", 1.64],
        ["C", "S", 1.92],
        ["C", "N", 1.57],
        ["C", "O", 1.57],
        ["N", "H", 1.11],
        ["N", "O", 1.5],
        ["N", "N", 1.55],
        ["N", "S", 2.06],
        ["O", "H", 1.07],
        ["O", "S", 1.52],
        ["O", "O", 1.58],
        ["S", "H", 1.45],
        ["S", "S", 2.17],
        ["H", "H", 0.84],
    ]

    etypes = [x.etype for x in atoms]
    datas = [x.data for x in atoms]
    target_etype = atom.etype
    target_data = atom.data
    bonds = []
    for x in atoms:
        etype = x.etype
        data = x.data
        for item in maximal_bond_lengths:
            if (etype == item[0] and target_etype == item[1]) or (
                etype == item[1] and target_etype == item[0]
            ):
                dist = math.dist(data, target_data)

                if dist < item[2]:
                    startpoint = target_data
                    endpoint = data
                    midpoint = (target_data + data) / 2
                    start_type = target_etype
                    end_type = etype

                    half_bond = Line(startpoint, midpoint, start_type, end_type)
                    bonds.append(half_bond)

    return bonds


def get_bonded_atoms(atoms):
    bonded_atoms = []
    for atom in atoms:
        bonds = get_bond(atoms, atom)
        bonded_atom = bonded_Atom(bonds, atom)
        bonded_atoms.append(bonded_atom)
    return bonded_atoms


def get_protein_struct_from_pdb(pdbid, filepath):
    parser = PDB.PDBParser()
    struct = parser.get_structure(pdbid, filepath)
    total_struct = []
    for model in struct:
        for chain in model:
            cur_struc = None
            pre_struc = None
            for residue in chain:
                atoms = []
                for elm in residue:
                    if elm.get_full_id()[3][0].strip() == "":
                        atom_type = "ATOM"
                    else:
                        atom_type = "HETATM"
                    if atom_type == "ATOM":
                        atom = Atom(
                            elm.get_coord(),
                            elm.element,
                            elm.fullname,
                            elm.get_serial_number(),
                        )
                        atoms.append(atom)

                # find all bonds inside a residue
                cur_struc = get_bonded_atoms(atoms)

                # add bond between adjecant residue
                if pre_struc != None:
                    for cur_bonded_atom in cur_struc:
                        if cur_bonded_atom.atom.eid == " N  ":
                            for pre_bonded_atom in pre_struc:
                                if pre_bonded_atom.atom.eid == " C  ":
                                    startpoint = pre_bonded_atom.atom.data
                                    endpoint = cur_bonded_atom.atom.data
                                    start_type = pre_bonded_atom.atom.etype
                                    end_type = cur_bonded_atom.atom.etype
                                    midpoint = (startpoint + endpoint) / 2

                                    line1 = Line(
                                        startpoint, midpoint, start_type, end_type
                                    )

                                    line2 = Line(
                                        endpoint, midpoint, end_type, start_type
                                    )

                                    pre_bonded_atom.add_line(line1)
                                    cur_bonded_atom.add_line(line2)

                ####
                total_struct += cur_struc
                pre_struc = cur_struc

    return total_struct


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
        if "ligand" in l and contents[idx - 1] == "@<TRIPOS>MOLECULE":
            natom, nbonds = contents[idx + 1].split()[:2]
            break

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

                if self.integral_type == "median":
                    res = np.median(L)
                elif self.integral_type == "std":
                    res = np.std(L)

                self.result[
                    bonded_atom1.atom.serial_id, bonded_atom2.atom.serial_id
                ] = res

    def get_result(self, bonded_atom1, bonded_atom2):
        return self.result[bonded_atom1.atom.serial_id, bonded_atom2.atom.serial_id]


def cutoff_struct(protein_struct, ligand_struct, cutoff=12):
    new_struct = []
    for bonded_atom1 in protein_struct:
        for bonded_atom2 in ligand_struct:
            atom1_pos = bonded_atom1.atom.data
            atom2_pos = bonded_atom2.atom.data
            if math.dist(atom1_pos, atom2_pos) < cutoff:
                new_struct.append(bonded_atom1)
                break
    return new_struct


def get_inteval(ep, el):

    scales = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    intevals = []
    if ep in el_p and el in el_l:
        for i in range(len(scales) - 1):
            r1 = scales[i]
            r2 = scales[i + 1]
            intevals.append([r1, r2])

    return intevals


def get_kda_feature_p(e1, e2, r1, r2, protein_struct, ligand_struct, F, args):
    Gli_sum = []
    for bonded_atom1 in protein_struct:
        if bonded_atom1.atom.etype == e1:
            gli_sum = 0
            for bonded_atom2 in ligand_struct:
                if bonded_atom2.atom.etype == e2:
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


def get_KDA_features(args, pdbid):
    filepath = "%s/%s/%s_protein.pdb" % (args.pdb_path, pdbid, pdbid)
    mol2file = "%s/%s/%s_ligand.mol2" % (args.pdb_path, pdbid, pdbid)

    protein_struct = get_protein_struct_from_pdb(pdbid, filepath)
    ligand_struct = get_ligand_struct_from_mol2(mol2file)
    protein_struct = cutoff_struct(protein_struct, ligand_struct)
    F = Gauss_linking_integral_of_structures(
        protein_struct, ligand_struct, args.integral_type
    )

    kda_feature = []
    for e1 in el_p:
        for e2 in el_l:
            for scale in get_inteval(e1, e2):
                r1, r2 = scale
                kda_feature_p = get_kda_feature_p(
                    e1, e2, r1, r2, protein_struct, ligand_struct, F, args
                )
                kda_feature_l = get_kda_feature_l(
                    e1, e2, r1, r2, protein_struct, ligand_struct, F, args
                )
                kda_feature += kda_feature_p
                kda_feature += kda_feature_l

    return kda_feature


def main():
    parser = argparse.ArgumentParser(description="Get KDA features for pdbbind")
    parser.add_argument("--pdb_path", type=str, default="datasets/PDBbind")
    parser.add_argument("--pdbid", type=str, default="2eg8")
    parser.add_argument("--bin_or_all", type=str, default="bin", help="bin or all")
    parser.add_argument(
        "--integral_type", type=str, default="median", help="median or std"
    )
    args = parser.parse_args()

    kda_feature = get_KDA_features(args, args.pdbid)

    print(np.shape(kda_feature))
    np.save(f"{args.pdbid}-complex-{args.integral_type}-{args.bin_or_all}", kda_feature)


if __name__ == "__main__":
    main()
    print("End!")
