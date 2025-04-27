#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:03:40 2022

@author: shenli
"""
import numpy as np
from scipy.spatial import distance_matrix
from Bio import PDB


def Gauss_linking_integral(a, b):
    """
    a:tuple; elements are head and tail
    of line segment, each is a (3,) array representing the xyz coordinate.
    """
    # a0, a1 = a[0],a[1]
    # b0, b1 = b[0],b[1]

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


def line_classification(cloudpoints, center_index, r):
    """
    cloudpoints:2d array: (m_samples,3), each row represents the xyz coordinate；
                line segment exists for each adjacent row.

    center_index: the index of point as center
    r: radiu of filtration
    """
    # find index of all points whose distance to center < r.
    distances = distance_matrix(cloudpoints, cloudpoints[center_index].reshape(1, -1))
    in_line = []
    out_line = []

    for i in range(cloudpoints.shape[0] - 1):
        if distances[i] < r:
            if distances[i + 1] < r:
                if i != center_index and i + 1 != center_index:
                    out_line.append((cloudpoints[i, :], cloudpoints[i + 1, :]))
                else:
                    in_line.append((cloudpoints[i, :], cloudpoints[i + 1, :]))

    return out_line, in_line


def line_classification1(cloudpoints, center_index, r):
    distances = distance_matrix(cloudpoints, cloudpoints[center_index].reshape(1, -1))
    in_line = []
    out_line = []
    if distances[0] < r:
        end2 = (cloudpoints[0] + cloudpoints[1]) / 2
        end1 = cloudpoints[0]
        if center_index == 0:
            in_line.append((end1, end2))
        else:
            out_line.append((end1, end2))

    for i in range(1, cloudpoints.shape[0] - 1):
        if distances[i] < r:
            if i != center_index:
                out_line.append(
                    ((cloudpoints[i - 1] + cloudpoints[i]) / 2, cloudpoints[i])
                )
                out_line.append(
                    (cloudpoints[i], (cloudpoints[i] + cloudpoints[i + 1]) / 2)
                )
            else:
                in_line.append(
                    ((cloudpoints[i - 1] + cloudpoints[i]) / 2, cloudpoints[i])
                )
                in_line.append(
                    (cloudpoints[i], (cloudpoints[i] + cloudpoints[i + 1]) / 2)
                )

    n = cloudpoints.shape[0] - 1

    if distances[n] < r:
        end1 = (cloudpoints[n - 1] + cloudpoints[n]) / 2
        end2 = cloudpoints[n]
        if center_index == n:
            in_line.append((end1, end2))
        else:
            out_line.append((end1, end2))

    return out_line, in_line


def get_protein_ca_atom_coordinate(pdbid, filepath):
    parser = PDB.PDBParser()
    struct = parser.get_structure(pdbid, filepath)

    CA_coordinates = np.array([])

    for model in struct:
        coor = []
        labels = np.array([])
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.id == "CA":
                        XYZ = atom.get_coord()
                        bfactor = atom.bfactor
                        labels = np.append(labels, bfactor)
                        CA_coordinates = np.hstack([CA_coordinates, XYZ])

        break

    return CA_coordinates.reshape(-1, 3), labels


def GLI_feature(CA_coor, r):
    GLI = []
    for i in range(CA_coor.shape[0]):
        outline, inline = line_classification1(CA_coor, i, r)
        L = 0

        for line1 in outline:
            for line2 in inline:
                l = Gauss_linking_integral(line1, line2)
                L += np.abs(l)

        if L == 0:
            GLI.append(0)
        else:
            GLI.append(1 / (L + 1e-4))

    return GLI


def GLI_feature_standard(CA_coor, r):
    GLI = []
    for i in range(CA_coor.shape[0]):
        outline, inline = line_classification1(CA_coor, i, r)
        L = 0

        for line1 in outline:
            for line2 in inline:
                l = Gauss_linking_integral(line1, line2)
                L += np.abs(l)

        GLI.append(L)

    return GLI
