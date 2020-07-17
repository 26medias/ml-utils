import os
import glob
import random
import math
import datetime as dt
import json
import ntpath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class featureReader:
  def __init__(self, groups=[], look_back=50, horizon=8, dataRange=(0,1), featureShape="matrix", cache=True):
    self.groups       = groups
    self.look_back    = look_back
    self.horizon      = horizon
    self.dataRange    = dataRange
    self.cache        = cache
    self.featureShape = featureShape
    self.dfCache      = {}
  
  def valmap(self, value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart))


  def dpct(self, start, end):
    return (end-start)/start;


  def load(self, glob_pattern):
    self.glob_pattern = glob_pattern


  def load_dataset(self, csv_filename):
    # Retrieve from the cache if there's one
    if self.cache and csv_filename in self.dfCache:
      return self.dfCache[csv_filename]
    # Open the csv file into a dataframe, remove the invalid rows
    df = pd.read_csv(csv_filename)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df = df.reset_index(drop=True)
    df['t'] = pd.to_datetime(df['t'], unit='s')
    df = df.sort_values('t')
    # Cache the data
    if self.cache:
      self.dfCache[csv_filename] = df
    return df


  def getFeatures(self):
    features = {}
    for groupName in self.groups:
      for dataset in glob.glob(self.glob_pattern):
        _features = self.getFeaturesForGroup(dataset, self.groups[groupName])
        if groupName not in features:
          features[groupName] = _features
        else:
          features[groupName] = np.append(features[groupName], _features, axis=0)
    return features
  
  
  def reshapeFeatures(self, features):
    if self.featureShape is "matrix":
      return features
    elif self.featureShape is "img":
      return features.reshape(features.shape[0], features.shape[1], 1)

  def getFeaturesForGroup(self, dataset, group):
    # Load the dataset
    df = self.load_dataset(dataset)
    # Create the subset dataframe
    _df = df[group['columns']].copy()
    for col in _df.columns:
      if col!='t' and col!='o':
        _df[col] = _df[col].transform(lambda x: self.valmap(x, group['inputRange'][0], group['inputRange'][1], self.dataRange[0], self.dataRange[1]))
    
    # Convert the dataframe to numpy
    lines = np.asarray(_df);

    # Assemble the timesteps
    timesteps_X = []
    l = len(lines)
    for n, v in enumerate(lines):
      if n >= self.look_back and n<l-self.horizon:
        in_steps = []
        for i in range(self.look_back-1, -1, -1):
          in_steps.append(np.array(lines[n-i]))
        
        _line = np.array(in_steps);
        _line = np.rot90(np.asarray(_line), 1)
        #print(">shape: ", _line.shape)
        #_line = _line.reshape(_line.shape[0], _line.shape[1], 1)
        _line = self.reshapeFeatures(_line)
        timesteps_X.append(_line)
    timesteps_X   = np.asarray(timesteps_X)
    timesteps_X   = np.clip(timesteps_X, a_min = self.dataRange[0], a_max = self.dataRange[1])

    return timesteps_X
  

  def getFeaturePrices(self):
    output = np.asarray([])
    for dataset in glob.glob(self.glob_pattern):
      df   = self.load_dataset(dataset)
      _dft = df[['t']].copy()
      _dfo = df['o'].tolist()
      
      # Convert the dataframe to numpy
      lines = np.asarray(_dfo);

      # Assemble the timesteps
      timesteps_X = []
      l = len(lines)
      for n, v in enumerate(lines):
        if n >= self.look_back and n<l-self.horizon:
          in_steps = []
          for i in range(self.look_back-1, -1, -1):
            in_steps.append(np.array(lines[n-i]))
          _line = np.array(in_steps)
          timesteps_X.append(np.asarray(_line))
      timesteps_X   = np.asarray(timesteps_X)
      if len(output)==0:
        output  = timesteps_X
      else:
        output  = np.append(output, timesteps_X, axis=0)

    return output
  
  def getHorizonOutput(self, in_steps, outputType):
    if outputType=="range":   
      outputLine = np.array([self.dpct(in_steps[0], min(in_steps)), self.dpct(in_steps[0], max(in_steps))])
    elif outputType=="signs":
      outputLine = np.array([1 if self.dpct(in_steps[0], min(in_steps))<0 else 0, 1 if self.dpct(in_steps[0], max(in_steps))>0 else 0])
    elif outputType=="diff":
      outputLine = ((np.array(in_steps)-in_steps[0])/in_steps[0])[1:]
    elif outputType=="count":
      _p = in_steps[0]
      outputLine = np.asarray([np.count_nonzero(np.array(in_steps)<_p)/(self.horizon-1), np.count_nonzero(np.array(in_steps)>_p)/(self.horizon-1)])
    else:
      outputLine = np.array(in_steps)
    return outputLine

  # range:    Range in percent
  # signs:    Signs within range
  def getTargets(self, outputType="range"):
    output = np.asarray([])
    for dataset in glob.glob(self.glob_pattern):
      df   = self.load_dataset(dataset)
      _dfo = df['o'].tolist()
      
      # Convert the dataframe to numpy
      lines = np.asarray(_dfo);

      # Assemble the timesteps
      timesteps_X = []
      l = len(lines)
      for n, v in enumerate(lines):
        if n >= self.look_back and n<l-self.horizon:
          in_steps = []
          for i in range(0, self.horizon, 1):
            in_steps.append(lines[n+i])
          outputLine = self.getHorizonOutput(in_steps, outputType)
          timesteps_X.append(outputLine)
      timesteps_X   = np.asarray(timesteps_X)
      if len(output)==0:
        output  = timesteps_X
      else:
        output  = np.append(output, timesteps_X, axis=0)
    return output


  def previewFeature(self, features, prices, idx=0):
    fig, axs = plt.subplots(len(features.keys())+1, figsize=(15,5))
    plt.autoscale(tight=True)
    fig.suptitle('Features Preview')
    for i, (k, v) in enumerate(features.items()):
      axs[i].imshow(features[k][idx].reshape(features[k][idx].shape[0], features[k][idx].shape[1]), cmap='RdBu', aspect="auto")
    axs[len(axs)-1].plot(prices[idx])
