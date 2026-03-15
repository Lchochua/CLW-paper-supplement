#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:06:12 2024

@author: lashachochua
"""

import numpy as np
import pandas as pd
import glob
import os
import time
import multiprocessing

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings('ignore', message='Intel MKL WARNING')
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

# Utility excel files from Mathematica
# Open all excel files of utilities under different policy regimes from working directory.
directory_path = "./"
excel_files = glob.glob(directory_path + "*.xlsx")
#print(excel_files)

matrices_dict = {}

for excel_file in excel_files:
    # Get the filename without the extension
    file_name = os.path.splitext(os.path.basename(excel_file))[0]

    # Load the Excel file into a pandas dataframe
    df = pd.read_excel(excel_file, header=None)

    # Convert the dataframe to a matrix
    matrix = df.values

    # Store the matrix in the dictionary with the same name as the Excel file
    matrices_dict[file_name] = matrix
    

# Settings of the model

# Define the set of players 'P' and coalitions 'S'
P = range(3)  # 0 = player 1, 1 = player 2, 2 = player 3
S = range(7)  # 0 = {1,2,3}, 1 = {1}, 2 = {2}, 3 = {3}, 4 = {1,2}, 5 = {2,3}, 6 = {3,1}

# Fix the number of maximal outcomes 'nmax'
nmax = 31  # 0 = MFN, 1 = CU(12), 2 = CU(23), 3 = CU(31), 4 = CU(123), 5 = FTA(12), 6 = FTA(23), 7 = FTA(31),
# 8 = FTA(12,31), 9 = FTA(12,23), 10 = FTA(23,31), 11 = FTA(123), 12 = PL(12), 13 = PL(23), 14 = PL(31),
# 15 = PL(123), 16 = FTAPL(12,31), 17 = FTAPL(12,23), 18 = FTAPL(23,12), 19 = FTAPL(23,31), 20 = FTAPL(31,12),
# 21 = FTAPL(31,23), 22 = FTAPL(12,123), 23 = FTAPL(23,123), 24 = FTAPL(31,123), 25 = HUBPL(12-31,23),
# 26 = HUBPL(12-23,31), 27 = HUBPL(23-31,12), 28 = CUPL(12,123), 29 = CUPL(23,123), 30 = CUPL(31,123)

n = 25 # Choose the number of firms in heterogeneous sector



# A.1 - Function 'elementwise': An element-wise application of an operator to two matrices 
#
# Input: Operator operator, 'm times n'-Matrix M, 'm times n'-Matrix N
# Output: 'm times n'-Matrix A
def elementwise(operator, M, N):
    assert(M.shape == N.shape)
    nr, nc = M.shape[0], M.shape[1]
    A =  np.zeros((nr,nc))
    for r in range(nr):
        for c in range(nc):
            A[r,c] = operator(M[r,c], N[r,c])    
    return np.matrix(A)




# A.2 - Function 'Network': Construct the network matrix for each coalition
#
# Input: Maximum Number nmax, Coalitions S
# Output: 'nmax times nmax'-Matrices A
def Network(nmax, S):
    # Step 1: Define useful arrays
    I = np.identity(nmax)
    z = np.zeros(nmax)
    # Step 2: Construct the network matrix for each coalition
    A = [np.matrix(np.identity(nmax)) for i in S]
    A[0] = np.matrix(np.ones((nmax, nmax)))
    A[1] = np.matrix(
        A[1] + [z, I[0], z, I[0], I[2], I[0], z, I[0], I[0] + I[5] + I[7], I[6], I[6], I[6] + I[9] + I[10], I[0], I[15],
                I[0], I[13], I[0] + I[5] + I[14], I[13] + I[15] + I[22], I[6], I[6], I[0] + I[7] + I[12], I[13] + I[15] + I[24],
                I[13] + I[15] + I[17], I[6], I[13] + I[15] + I[21], I[13] + I[15] + I[17] + I[21] + I[22] + I[24], I[6] + I[9] + I[19],
                I[6] + I[10] + I[18], I[1] + I[13] + I[15], I[2], I[3] + I[13] + I[15]])
    A[2] = np.matrix(
        A[2] + [z, I[0], I[0], z, I[3], I[0], I[0], z, I[7], I[0] + I[5] + I[6], I[7], I[7] + I[8] + I[10], I[0], I[0],
                I[15], I[14], I[14] + I[15] + I[22], I[0] + I[5] + I[13], I[0] + I[6] + I[12], I[14] + I[15] + I[23],
                I[7], I[7], I[14] + I[15] + I[16], I[14] + I[15] + I[19], I[7], I[7] + I[8] + I[21], I[14] + I[15] + I[16] + I[19] + I[22] + I[23],
                I[7] + I[10] + I[20], I[1] + I[14] + I[15], I[2] + I[14] + I[15], I[3]])
    A[3] = np.matrix(
        A[3] + [z, z, I[0], I[0], I[1], z, I[0], I[0], I[5], I[5], I[0] + I[6] + I[7], I[5] + I[8] + I[9], I[15], I[0],
                I[0], I[12], I[5], I[5], I[12] + I[15] + I[23], I[0] + I[6] + I[14], I[12] + I[15] + I[24], I[0] + I[7] + I[13],
                I[5], I[12] + I[15] + I[18], I[12] + I[15] + I[20], I[5] + I[8] + I[17], I[5] + I[9] + I[16],
                I[12] + I[15] + I[18] + I[20] + I[23] + I[24], I[1], I[2] + I[12] + I[15], I[3] + I[12] + I[15]])
    A[4] = np.matrix(
        A[4] + [I[1] + I[5] + I[12], 
                I[0] + I[5] + I[12], 
                I[0] + I[1] + I[5] + I[12], 
                I[0] + I[1] + I[5] + I[12],
                I[0] + I[1] + I[2] + I[3] + I[5] + I[12], 
                I[0] + I[1] + I[12], 
                I[0] + I[1] + I[5] + I[9] + I[12] + I[18],
                I[0] + I[1] + I[5] + I[8] + I[12] + I[20], 
                I[0] + I[1] + I[5] + I[7] + I[12] + I[20], 
                I[0] + I[1] + I[5] + I[6] + I[12] + I[18],
                I[0] + I[1] + I[5] + I[6] + I[7] + I[8] + I[9] + I[11] + I[12] + I[18] + I[20] + I[27], 
                I[0] + I[1] + I[5] + I[6] + I[7] + I[8] + I[9] + I[10] + I[12] + I[18] + I[20] + I[27],
                I[0] + I[1] + I[5], 
                I[0] + I[1] + I[5] + I[12] + I[14] + I[15] + I[16] + I[17] + I[22] + I[28], 
                I[0] + I[1] + I[5] + I[12] + I[13] + I[15] + I[16] + I[17] + I[22] + I[28],
                I[0] + I[1] + I[5] + I[12] + I[13] + I[14] + I[16] + I[17] + I[22] + I[28], 
                I[0] + I[1] + I[5] + I[12] + I[13] + I[14] + I[15] + I[17] + I[22] + I[28],
                I[0] + I[1] + I[5] + I[12] + I[13] + I[14] + I[15] + I[16] + I[22] + I[28],
                I[0] + I[1] + I[5] + I[6] + I[9] + I[12],
                I[0] + I[1] + I[5] + I[6] + I[9] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[18] + I[22] + I[23] + I[26] + I[28], 
                I[0] + I[1] + I[5] + I[7] + I[8] + I[12],
                I[0] + I[1] + I[5] + I[7] + I[8] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[20] + I[22] + I[24] + I[25] + I[28], 
                I[0] + I[1] + I[5] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[28],
                I[0] + I[1] + I[5] + I[6] + I[9] + I[12]+ I[13] + I[14] + I[15] + I[16] + I[17] + I[19] + I[22] + I[26] + I[28],
                I[0] + I[1] + I[5] + I[7] + I[8] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[21] + I[22] + I[25] + I[28],
                I[0] + I[1] + I[5] + I[7] + I[8] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[20] + I[21] + I[22] + I[24] + I[28],
                I[0] + I[1] + I[5] + I[6] + I[9] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[18] + I[19] + I[22] + I[23] + I[28],
                I[0] + I[1] + I[5] + I[6] + I[7] + I[8] + I[9] + I[10] + I[11] + I[12] + I[18] + I[20],
                I[0] + I[1] + I[5] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[22],
                I[0] + I[1] + I[2] + I[5] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[22] + I[28],
                I[0] + I[1] + I[3] + I[5] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[22] + I[28]])
    A[5] = np.matrix(
        A[5] + [I[2] + I[6] + I[13],
                I[0] + I[2] + I[6] + I[13],
                I[0] + I[6] + I[13],
                I[0] + I[2] + I[6] + I[13],
                I[0] + I[1] + I[2] + I[3] + I[6] + I[13],
                I[0] + I[2] + I[6] + I[9] + I[13] + I[17],
                I[0] + I[2] + I[13],
                I[0] + I[2] + I[6] + I[10] + I[13] + I[21],
                I[0] + I[2] + I[5] + I[6] + I[7] + I[9] + I[10] + I[11] + I[13] + I[17] + I[21] + I[25],
                I[0] + I[2] + I[5] + I[6] + I[13] + I[17],
                I[0] + I[2] + I[6] + I[7] + I[13] + I[21],
                I[0] + I[2] + I[5] + I[6] + I[7] + I[8] + I[9] + I[10] + I[13] + I[17] + I[21] + I[25],
                I[0] + I[2] + I[6] + I[13] + I[14] + I[15] + I[18] + I[19] + I[23] + I[29],
                I[0] + I[2] + I[6],
                I[0] + I[2] + I[6] + I[12] + I[13] + I[15] + I[18] + I[19] + I[23] + I[29], 
                I[0] + I[2] + I[6] + I[12] + I[13] + I[14] + I[18] + I[19] + I[23] + I[29],
                I[0] + I[2] + I[5] + I[6] + I[9] + I[12] + I[13] + I[14] + I[15] + I[17] + I[18] + I[19] + I[22] + I[23] + I[26] + I[29],
                I[0] + I[2] + I[5] + I[6] + I[9] + I[13],
                I[0] + I[2] + I[6] + I[12] + I[13] + I[14] + I[15] + I[19] + I[23] + I[29],
                I[0] + I[2] + I[6] + I[12] + I[13] + I[14] + I[15] + I[18] + I[23] + I[29],
                I[0] + I[2] + I[6] + I[7] + I[10] + I[12] + I[13] + I[14] + I[15] + I[18] + I[19] + I[21] + I[23] + I[24] + I[27] + I[29],
                I[0] + I[2] + I[6] + I[7] + I[10] + I[13], 
                I[0] + I[2] + I[5] + I[6] + I[9] + I[12] + I[13] + I[14] + I[15] + I[16] + I[18] + I[19] + I[23] + I[26] + I[29],
                I[0] + I[2] + I[6] + I[12] + I[13] + I[14] + I[15] + I[18] + I[19] + I[29], 
                I[0] + I[2] + I[6] + I[7] + I[10] + I[12] + I[13] + I[14] + I[15] + I[18] + I[19] + I[20] + I[23] + I[27] + I[29],
                I[0] + I[2] + I[5] + I[6] + I[7] + I[8] + I[9] + I[10] + I[11] + I[13] + I[17] + I[21],
                I[0] + I[2] + I[5] + I[6] + I[9] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[18] + I[19] + I[22] + I[23] + I[29],
                I[0] + I[2] + I[6] + I[7] + I[10] + I[12] + I[13] + I[14] + I[15] + I[18] + I[19] + I[20] + I[21] + I[23] + I[24] + I[29],
                I[0] + I[1] + I[2] + I[6] + I[12] + I[13] + I[14] + I[15] + I[18] + I[19] + I[23] + I[29],
                I[0] + I[2] + I[6] + I[12] + I[13] + I[14] + I[15] + I[18] + I[19] + I[23],
                I[0] + I[2] +  I[3] + I[6] + I[12] + I[13] + I[14] + I[15] + I[18] + I[19] + I[23] + I[29]])
    A[6] = np.matrix(
        A[6] + [I[3] + I[7] + I[14],
                I[0] + I[3] + I[7] + I[14],
                I[0] + I[3] + I[7] + I[14],
                I[0] + I[7] + I[14],
                I[0] + I[1] + I[2] + I[3] + I[7] + I[14],
                I[0] + I[3] + I[7] + I[8] + I[14] + I[16],
                I[0] + I[3] + I[7] + I[10] + I[14] + I[19],
                I[0] + I[3] + I[14],
                I[0] + I[3] + I[5] + I[7] + I[14] + I[16],
                I[0] + I[3] + I[5] + I[6] + I[7] + I[8] + I[10] + I[11] + I[14] + I[16] + I[19] + I[26],
                I[0] + I[3] + I[6] + I[7] + I[14] + I[19],
                I[0] + I[3] + I[5] + I[6] + I[7] + I[8] + I[9] + I[10] + I[14] + I[16] + I[19] + I[26],
                I[0] + I[3] + I[7] + I[13] + I[14] + I[15] + I[20] + I[21] + I[24] + I[30], 
                I[0] + I[3] + I[7] + I[12] + I[14] + I[15] + I[20] + I[21] + I[24] + I[30],
                I[0] + I[3] + I[7],
                I[0] + I[3] + I[7] + I[12] + I[13] + I[14] + I[20] + I[21] + I[24] + I[30], 
                I[0] + I[3] + I[5] + I[7] + I[8] + I[14],
                I[0] + I[3] + I[5] + I[7] + I[8] + I[12] + I[13] + I[14] + I[15] + I[16] + I[20] + I[21] + I[22] + I[24] + I[25] + I[30],
                I[0] + I[3] + I[6] + I[7] + I[10] + I[12] + I[13] + I[14] + I[15] + I[19] + I[20] + I[21] + I[23] + I[24] + I[27] + I[30], 
                I[0] + I[3] + I[6] + I[7] + I[10] + I[14],
                I[0] + I[3] + I[7] + I[12] + I[13] + I[14] + I[15] + I[21] + I[24] + I[30], 
                I[0] + I[3] + I[7] + I[12] + I[13] + I[14] + I[15] + I[20] + I[24] + I[30],
                I[0] + I[3] + I[5] + I[7] + I[8] + I[12] + I[13] + I[14] + I[15] + I[17] + I[20] + I[21] + I[24] + I[25] + I[30],
                I[0] + I[3] + I[6] + I[7] + I[10] + I[12] + I[13] + I[14] + I[15] + I[18] + I[20] + I[21] + I[24] + I[27] + I[30], 
                I[0] + I[3] + I[7] + I[12] + I[13] + I[14] + I[15] + I[20] + I[21] + I[30],
                I[0] + I[3] + I[5] + I[7] + I[8] + I[12] + I[13] + I[14] + I[15] + I[16] + I[17] + I[20] + I[21] + I[22] + I[24] + I[30],
                I[0] + I[3] + I[5] + I[6] + I[7] + I[8] + I[9] + I[10] + I[11] + I[14] + I[16] + I[19],
                I[0] + I[3] + I[6] + I[7] + I[10] + I[12] + I[13] + I[14] + I[15] + I[18] + I[19]+ I[20] + I[21] + I[23] + I[24] + I[30],
                I[0] + I[1] + I[3] + I[7] + I[12] + I[13] + I[14] + I[15] + I[20] + I[21] + I[24] + I[30],
                I[0] + I[2] + I[3] + I[7] + I[12] + I[13] + I[14] + I[15] + I[20] + I[21] + I[24] + I[30],
                I[0] + I[3] + I[7] + I[12] + I[13] + I[14] + I[15] + I[20] + I[21] + I[24]])
    return A



# A.3 - Function 'Pref': Compute the utility matrix for each player and consequentially determine the preference matrix for each coalition
#    
# Input: Players P, Coalitions S, Outcomes X, k and m are rows and columns of the relevant utility matrix (corresponding different values of parameters of the model)
# Output: Utility Matrices U, Preference Matrices B
# Country 1 is large and countries 2 and 3 are small

def Pref(P,S,X,k,m,n):
    # Step 1: Utilities 
    Util = [[0 for x in X] for p in P]
    for p in P:
        # Case 0: MFN   
        pme = ((p) % 3)
        entry = 0
        if p == 0:
            Util[pme][entry] = matrices_dict[f"MFN_{n}"][k,m] + matrices_dict["Tr_MFN_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"MFN_{n}"][k,m] + matrices_dict["Tr_MFN_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"MFN_{n}"][k,m] + matrices_dict["Tr_MFN_S"][k,m]
        
        # Cases 1: CU(12)
        pme = ((p) % 3)
        entry = 1
        if p == 0:
            Util[pme][entry] = matrices_dict[f"CUI_{n}"][k,m] + matrices_dict["Tr_CU_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"CUI_{n}"][k,m] + matrices_dict["Tr_CU_sl_SI"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"CUO_{n}"][k,m] + matrices_dict["Tr_CU_sl_SO"][k,m]
        
        # Cases 2: CU(23)
        pme = ((p) % 3)
        entry = 2
        if p == 0:
            Util[pme][entry] = matrices_dict[f"CUO_{n}"][k,m] + matrices_dict["Tr_CU_ss_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"CUI_{n}"][k,m] + matrices_dict["Tr_CU_ss_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"CUI_{n}"][k,m] + matrices_dict["Tr_CU_ss_S"][k,m]
        
        # Cases 3: CU(31)
        pme = ((p) % 3)
        entry = 3
        if p == 0:
            Util[pme][entry] = matrices_dict[f"CUI_{n}"][k,m] + matrices_dict["Tr_CU_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"CUO_{n}"][k,m] + matrices_dict["Tr_CU_sl_SO"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"CUI_{n}"][k,m] + matrices_dict["Tr_CU_sl_SI"][k,m]
        
        # Cases 5: FTA(12)
        pme = ((p) % 3)
        entry = 5
        if p == 0:
            Util[pme][entry] = matrices_dict[f"FTAI_{n}"][k,m] + matrices_dict["Tr_FTA_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"FTAI_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SI"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"FTAO_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SO"][k,m]
        
        
        # Cases 6: FTA(23)
        pme = ((p) % 3)
        entry = 6
        if p == 0:
            Util[pme][entry] = matrices_dict[f"FTAO_{n}"][k,m] + matrices_dict["Tr_FTA_ss_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"FTAI_{n}"][k,m] + matrices_dict["Tr_FTA_ss_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"FTAI_{n}"][k,m] + matrices_dict["Tr_FTA_ss_S"][k,m]
        
        # Cases 7: FTA(31)
        pme = ((p) % 3)
        entry = 7
        if p == 0:
            Util[pme][entry] = matrices_dict[f"FTAI_{n}"][k,m] + matrices_dict["Tr_FTA_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"FTAO_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SO"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"FTAI_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SI"][k,m]
        
        # Cases 8: FTAHUB(1)
        pme = ((p) % 3)
        entry = 8
        if p == 0:
            Util[pme][entry] = matrices_dict[f"HUB_{n}"][k,m] + matrices_dict["Tr_HUB_l_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"SPOKE_{n}"][k,m] + matrices_dict["Tr_HUB_l_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"SPOKE_{n}"][k,m] + matrices_dict["Tr_HUB_l_S"][k,m]
        
        # Cases 9: FTAHUB(2)
        pme = ((p) % 3)
        entry = 9
        if p == 0:
            Util[pme][entry] = matrices_dict[f"SPOKE_{n}"][k,m] + matrices_dict["Tr_HUB_s_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"HUB_{n}"][k,m] + matrices_dict["Tr_HUB_s_SI"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"SPOKE_{n}"][k,m] + matrices_dict["Tr_HUB_s_SO"][k,m]
        
        # Cases 10: FTAHUB(3)
        pme = ((p) % 3)
        entry = 10
        if p == 0:
            Util[pme][entry] = matrices_dict[f"SPOKE_{n}"][k,m] + matrices_dict["Tr_HUB_s_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"SPOKE_{n}"][k,m] + matrices_dict["Tr_HUB_s_SO"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"HUB_{n}"][k,m] + matrices_dict["Tr_HUB_s_SI"][k,m]
        
        # Cases 12: PL(12)
        pme = ((p) % 3)
        entry = 12
        if p == 0:
            Util[pme][entry] = matrices_dict[f"PLI_{n}"][k,m] + matrices_dict["Tr_MFN_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"PLI_{n}"][k,m] + matrices_dict["Tr_MFN_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"PLO_{n}"][k,m] + matrices_dict["Tr_MFN_S"][k,m]
        
        # Cases 13: PL(23)
        pme = ((p) % 3)
        entry = 13
        if p == 0:
            Util[pme][entry] = matrices_dict[f"PLO_{n}"][k,m] + matrices_dict["Tr_MFN_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"PLI_{n}"][k,m] + matrices_dict["Tr_MFN_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"PLI_{n}"][k,m] + matrices_dict["Tr_MFN_S"][k,m]    
        
        # Cases 14: PL(31)
        pme = ((p) % 3)
        entry = 14
        if p == 0:
            Util[pme][entry] = matrices_dict[f"PLI_{n}"][k,m] + matrices_dict["Tr_MFN_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"PLO_{n}"][k,m] + matrices_dict["Tr_MFN_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"PLI_{n}"][k,m] + matrices_dict["Tr_MFN_S"][k,m]    
        
        # Cases 15: PL(123)
        pme = ((p) % 3)
        entry = 15
        if p == 0:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_MFN_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_MFN_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_MFN_S"][k,m]    
        
        
        # Case 16 FTAPL(12,31)
        pme = ((p) % 3)
        entry = 16
        if p == 0:
            Util[pme][entry] = matrices_dict[f"aFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"bFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SI"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"cFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SO"][k,m]
            
        # Case 17 FTAPL(12,23)
        pme = ((p) % 3)
        entry = 17
        if p == 0:
            Util[pme][entry] = matrices_dict[f"bFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"aFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SI"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"cFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SO"][k,m]
            
        # Case 18 FTAPL(23,12)
        pme = ((p) % 3)
        entry = 18
        if p == 0:
            Util[pme][entry] = matrices_dict[f"cFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_ss_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"aFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_ss_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"bFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_ss_S"][k,m]
            
        # Case 19 FTAPL(23,31)
        pme = ((p) % 3)
        entry = 19
        if p == 0:
            Util[pme][entry] = matrices_dict[f"cFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_ss_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"bFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_ss_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"aFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_ss_S"][k,m]
            
        # Case 20 FTAPL(31,12)
        pme = ((p) % 3)
        entry = 20
        if p == 0:
            Util[pme][entry] = matrices_dict[f"aFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"cFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SO"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"bFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SI"][k,m]
            
        # Case 21 FTAPL(31,23)
        pme = ((p) % 3)
        entry = 21
        if p == 0:
            Util[pme][entry] = matrices_dict[f"bFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"cFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SO"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"aFTAPL_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SI"][k,m]
            
        # Case 22 FTAPL(12,123)
        pme = ((p) % 3)
        entry = 22
        if p == 0:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_FTA_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SI"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SO"][k,m]
            
        # Case 23 FTAPL(23,123)
        pme = ((p) % 3)
        entry = 23
        if p == 0:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_FTA_ss_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_FTA_ss_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_FTA_ss_S"][k,m]    
            
        # Case 24 FTAPL(31,123)
        pme = ((p) % 3)
        entry = 24
        if p == 0:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_FTA_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SO"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_FTA_sl_SI"][k,m]   
        
        # Cases 25: HUBPL(12-31,23)
        pme = ((p) % 3)
        entry = 25
        if p == 0:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_HUB_l_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_HUB_l_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_HUB_l_S"][k,m]
        
        # Cases 26: HUBPL(12-23,31)
        pme = ((p) % 3)
        entry = 26
        if p == 0:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_HUB_s_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_HUB_s_SI"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_HUB_s_SO"][k,m]
        
        # Cases 27: HUBPL(23-31,12)
        pme = ((p) % 3)
        entry = 27
        if p == 0:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_HUB_s_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_HUB_s_SO"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_HUB_s_SI"][k,m]
        
        
        # Case 28 CUPL(12,123)
        pme = ((p) % 3)
        entry = 28
        if p == 0:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_CU_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_CU_sl_SI"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_CU_sl_SO"][k,m]
            
        # Case 29 CUPL(23,123)
        pme = ((p) % 3)
        entry = 29
        if p == 0:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_CU_ss_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_CU_ss_S"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_CU_ss_S"][k,m]    
            
        # Case 30 CUPL(31,123)
        pme = ((p) % 3)
        entry = 30
        if p == 0:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_CU_sl_L"][k,m]
        elif p == 1:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_CU_sl_SO"][k,m]
        else:
            Util[pme][entry] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_CU_sl_SI"][k,m]   
        
        # Cases 4,11,31: GFT
        pme = (p)
        #po1 = ((p+1) % 3)
        #po2 = ((p+2) % 3)
        entryCU = 4
        entryFTA = 11

        # Utilities for GFTCU [Case 4]        
        if p == 0:
            Util[pme][entryCU] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_GFT_L"][k,m]
        elif p == 1:
            Util[pme][entryCU] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_GFT_S"][k,m]
        else:
            Util[pme][entryCU] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_GFT_S"][k,m]
        # Utilities for GFTFTA [Case 11]        
        if p == 0:
            Util[pme][entryFTA] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_GFT_L"][k,m]
        elif p == 1:
            Util[pme][entryFTA] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_GFT_S"][k,m]
        else:
            Util[pme][entryFTA] = matrices_dict[f"GFT_{n}"][k,m] + matrices_dict["Tr_GFT_S"][k,m]
        
    # Step 2: Preferences
    B = [np.matrix(np.zeros((len(X),len(X)))) for s in S]
    for p in P:
        for i in X:
            for j in X:
                B[p+1][i,j]= np.greater(Util[p][j],Util[p][i])            
    B[4] = elementwise(np.multiply, B[1], B[2])
    B[5] = elementwise(np.multiply, B[2], B[3])
    B[6] = elementwise(np.multiply, B[3], B[1])
    B[0] = elementwise(np.multiply, B[4], B[3])
    # Step 3: Utilities with Preferences
    R = [Util,B]
    return R

    
# ----------------------------------------------------------------------------- 

# A.4 - Function 'Search': Finds the Stable Set (where the stability depends on a parameter)
#    
# Input: Coalitions S, Outcomes X, Network A, Preferences B, Parameter o 
# Output: Stable Set x
#
# Note: The parameter 'o' determines whether the search algorithm considers
# deviations from deviations with infinite (intermediate) steps (o = 0) or
# with o (intermediate) steps (o >= 1) or computes the undominated outcomes
# under direct dominance (o = -1)
def Search(S,X,A,B,o):
    # Step 1: 'Direct Dominance'
    C = [np.matrix(np.zeros((len(X),len(X)))) for s in S]
    for s in S:
        C[s] = elementwise(np.multiply, A[s], B[s])
    D = np.matrix(np.zeros((len(X),len(X))))
    for s in S:
        D = elementwise(np.logical_or, D, C[s])
    # Step 2: 'Indirect Dominance(s)'
    if o >= 0:
        if o >= 1:
            ocount = o
        if o == 0:
            ocount = 1
        while ocount >= 1:
            AD = [np.matrix(np.zeros((len(X),len(X)))) for s in S]
            E = [np.matrix(np.zeros((len(X),len(X)))) for s in S]
            for s in S:
                AD[s] = np.matrix([[1-np.prod([(1-A[s][i,z]*D[z,j]) for z in X]) for j in X] for i in X])
                E[s] = elementwise(np.multiply, elementwise(np.logical_or, A[s], AD[s]), B[s])
            F = np.matrix(np.zeros((len(X),len(X))))
            for s in S:
                F = elementwise(np.logical_or, F, E[s])
            if (F==D).all():
                break
            else: 
                D = F
                if o >= 1:
                    ocount = ocount - 1
    # Step 3: Find the Stable Set
    x = np.ones(len(X))
    xnext = np.zeros(len(X))
    if o >= 0:
        while True:
            for i in X:
                if x[i] == 0:
                    xnext[i] = 0
                else:
                    xneg = 0
                    for l in S:
                        for k in X:
                            xneghelp = A[l][i,k]*np.prod([(1-x[z]*(np.logical_or(np.equal(k,z),D[k,z]))*(1-B[l][i,z])) for z in X])
                            xneg = np.logical_or(xneg,xneghelp)
                    xnext[i] = x[i]-xneg
            if (xnext==x).all():
                break
            else:
                x = xnext
        return [x,D]
    if o == -1:
        for i in X:
            xneg = 0
            for j in X:
                xneg = np.logical_or(xneg,D[i,j])
            x[i] = x[i]-xneg
        return [x,D]


# Function to process the full parameter set
# i stands for sigma, and j stands for alpha 
def process_parameters(i_j):
    i, j = i_j
    LCS = Search(S, range(nmax), Network(nmax, S), Pref(P, S, range(nmax), i, j, n)[1], 0)
    lcs = LCS[0]
    return [i, j] + lcs.tolist()

# Main execution function
def main():
    parameters = [(i, j) for i in range(496) for j in range(451)]
    
    # Use multiprocessing Pool to handle parallel processing
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(process_parameters, parameters)
    
    # Save results to DataFrame
    columns = ['i', 'j'] + list(range(len(results[0]) - 2))
    df = pd.DataFrame(results, columns=columns)
    df.to_excel(f'ssl_stable_sets_31_regimes_full_{n}.xlsx', index=False)
    print('The End')

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")
    
    