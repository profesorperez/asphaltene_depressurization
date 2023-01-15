#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 18:11:00 2023

Average beads distance

@author: fperez
"""

#%% Import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Functions

def read_file(file_in):
  """
  Read file

  Parameters
  ----------
  file_in : string
    File (path and name) to be read

  Returns
  -------
  temp_data : list
    List of lists with the words in every line of the file read

  """

  temp_data = []

  f = open(file_in)

  for line in f:
    temp_data.append(line.split())

  f.close()

  return temp_data

def read_dump_file(file_in):
  """
  Read dump file from LAMMPS

  Parameters
  ----------
  file_in : string
    File (name and location) of dump file to be read

  Returns
  -------
  labels : list
    List of strings with labels
  box : numpy array
    Three dimensional numpy array (Ntimesteps x 3 x 2) with box dimensions (lo/hi)
  data_array : numpy array
    Three-dimensional numpy array (Ntimesteps x Nparticles x Nlabels). The Nlabels
    values are given in the line ITEM: ATOMS ...

  """
  
  data = read_file(file_in)
  
  numberOfParticles = int(data[3][0])
  offset = numberOfParticles + 9
  numberOfTimesteps = int(len(data)/offset)
  labels = data[8][2:]

  box = np.zeros((numberOfTimesteps, 3, 2))  
  data_array = np.zeros((numberOfTimesteps, numberOfParticles, len(labels)))
  m = 0
  for i in range(numberOfTimesteps):
    for k in range(3):
      box[i,k] = list(map(float, data[5+k+m*offset]))
    for j in range(numberOfParticles):
      data_array[i,j] = list(map(float, data[9+j+m*offset]))
    m += 1
  
  return(labels, box, data_array)

def distance(r1, r2, L):
  """
  Caclulate minimum distance in periodic boxes

  Parameters
  ----------
  r1 : array
    Vector with coordinates of particle 1
  r2 : array
    Vector with coordinates of particle 2
  L : array
    Vector with box dimensions

  Returns
  -------
  d : scalar
    Minimum distance between particles 1 and 2

  """

  dr = abs(r1-r2)
  dR = L - dr

  minr=[]
  for i in range(3):
    if dr[i] < dR[i]:
      minr.append(dr[i])
    else:
      minr.append(dR[i])

  return(np.linalg.norm(minr))

#%% Reference values for bonds and angles

reference_bond = {1 : 3.37698,\
                  2 : 3.82959,\
                  3 : 4.2822 ,\
                  4 : 3.86299,\
                  5 : 4.3135 ,\
                  6 : 4.278  ,\
                  7 : 3.56449}

reference_angle = {1 : 180     ,\
                   2 : 120     ,\
                   3 : 176.162 ,\
                   4 : 172.323 ,\
                   5 : 116.162 ,\
                   6 :  63.8383,\
                   7 : 150     ,\
                   8 : 157.6   ,\
                   9 : 90}

# Bonds formed in molecule 1 (they repeat for all other molecules)

## Format : [bond type, bead 1, bead 2]

bonds = np.array(
        [[1, 1, 10],\
         [1, 1, 6],\
         [1, 1, 2],\
         [1, 2, 10],\
         [1, 2, 6],\
         [1, 2, 7],\
         [1, 2, 3],\
         [1, 3, 8],\
         [1, 3, 7],\
         [1, 3, 4],\
         [1, 6, 12],\
         [1, 6, 7],\
         [1, 7, 8],\
         [1, 7, 12],\
         [7, 12, 20],\
         [1, 4, 11],\
         [1, 4, 9],\
         [1, 4, 5],\
         [1, 4, 8],\
         [1, 5, 11],\
         [1, 5, 9],\
         [4, 5, 15],\
         [1, 8, 9],\
         [2, 8, 14],\
         [1, 9, 13],\
         [2, 9, 14],\
         [4, 11, 19],\
         [3, 13, 14],\
         [5, 15, 16],\
         [6, 16, 17],\
         [5, 17, 18]])

# Angles formed in molecule 1 (they repeat for all other molecules)

## Format : [angle type, bead 1, bead 2, bead 3]

angles = np.array(
          [[1, 1, 2, 3],\
          [1, 7, 2, 10],\
          [2, 3, 2, 10],\
          [1, 2, 3, 4],\
          [1, 1, 6, 12],\
          [1, 6, 7, 8],\
          [1, 3, 7, 12],\
          [7, 7, 12, 20],\
          [9, 6, 12, 20],\
          [1, 3, 4, 5],\
          [1, 8, 4, 11],\
          [2, 3, 4, 11],\
          [7, 9, 5, 15],\
          [9, 11, 5, 15],\
          [1, 7, 8, 9],\
          [3, 3, 8, 14],\
          [5, 7, 8, 14],\
          [4, 4, 9, 13],\
          [3, 5, 9, 14],\
          [6, 13, 9, 14],\
          [7, 4, 11, 19],\
          [9, 5, 11, 19],\
          [8, 5, 15, 16],\
          [8, 15, 16, 17],\
          [8, 16, 17, 18]])
  
#%% Main

# Get data

path = '../data/'
pressure = [50, 75, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 300, 400, 700]
npress = len(pressure)

ip = 0
for p in pressure[-1:]:
  file_in = path + str(p) + '_atm/5-production_npt/clusters.dat'
  labels, box, data = read_dump_file(file_in)
  Ntimesteps, Nparticles, Nlabels = data.shape
  
  # Atom id index
  iid = labels.index('id')
  
  # Molecule index
  imol = labels.index('mol')
  
  # Atom type index
  iatype = labels.index('type')

  # Coordinates index
  ix = labels.index('x')
  iy = labels.index('y')
  iz = labels.index('z')

  for it in range(Ntimesteps):
    # Gather particles by molecule id
    molecules = {}
    for line in data[it]:
      molid = int(line[imol])
      if molid not in list(molecules.keys()):
        molecules[molid] = [line]
      else:
        molecules[molid].append(line)

    # Number of molecules
    Nmolecules = len(molecules)
    
    if it == 0:
      # Number of beads per molecule
      npm = int(Nparticles/Nmolecules)
      # Need to store deformations
      stretch_vec = np.zeros((npress, Ntimesteps, Nmolecules, len(bonds)))
      bend_vec = np.zeros((npress, Ntimesteps, Nmolecules, len(angles)))
      
    # Box dimensions at current timestep
    L = box[it][:,1] - box[it][:,0]

    
    for i in molecules.keys():
      # Bond deformation
      stretch = []
      for t, b1, b2 in bonds:
        p1 = molecules[i][b1-1][[ix, iy, iz]]
        p2 = molecules[i][b2-1][[ix, iy, iz]]
        d12 = distance(p1, p2, L)
        stretch.append(d12 - reference_bond[t])
      stretch_vec[ip, it, i-1, :] = stretch
    
      # Angle deformation
      bend = []
      for t, b1, b2, b3 in angles:
        p1 = molecules[i][b1-1][[ix, iy, iz]]
        p2 = molecules[i][b2-1][[ix, iy, iz]]
        p3 = molecules[i][b3-1][[ix, iy, iz]]
        d12 = distance(p1, p2, L)
        bend.append(d12 - reference_angle[t])
      bend_vec[ip, it, i-1, :] = bend