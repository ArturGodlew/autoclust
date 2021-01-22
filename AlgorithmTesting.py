from autoclust import auto_clust
from autoclust import P
from scipy.io import arff
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sklearn.mixture import GaussianMixture
import sklearn.cluster as cl
from sklearn import metrics
import collections
import pickle
import os.path
import importlib
import math
import random

class LabledPoint:
  def __init__(self, x, y, label):
    self.x = x
    self.y = y
    self.label = label


def KMeansAlg(clustersN, points):
    points['label'] = cl.KMeans(n_clusters=clustersN, random_state=0).fit(points).labels_
    return points

def SpectralClusteringAlg(clustersN, points):
    points['label'] = cl.SpectralClustering(n_clusters=clustersN, assign_labels="discretize", random_state=0).fit(points).labels_
    return points

def BirchAlg(clustersN, points):
    points['label'] = cl.Birch(n_clusters=clustersN).fit(points).labels_
    return points


def MeanShiftAlg(clustersN, points):
    points['label'] = cl.MeanShift(bandwidth=clustersN).fit(points).labels_
    return points

def OPTICSAlg(points):
    points['label'] = cl.OPTICS().fit(points).labels_
    return points

def AffinityPropagationAlg(points):
    points['label'] = cl.AffinityPropagation(random_state=0).fit(points).labels_
    return points

def AgglomerativeClusteringAlg(clustersN, points):
    points['label'] = cl.AgglomerativeClustering(n_clusters=clustersN).fit(points).labels_
    return points

def DBSCANAlg(points, labels, runs):
    bestScore = 0
    bestResult = 0
    bestEps = 0 
    for i in range(1, 1 + runs):
        result = cl.DBSCAN(eps=2*i).fit(points).labels_
        score = metrics.adjusted_rand_score(result, labels)
        if score > bestScore:
            bestResult = result
            bestScore = score
    points["label"] = result
    return points

def saveResults(saveFile, datasetNo, results, saveSubFolder, eps = None, scored = True):
    with open(saveFile, 'wb') as output:
        pickle.dump((datasetNo, results), output, pickle.HIGHEST_PROTOCOL)

    names = []
    scores = []
    for i in range(0,len(results)):
        points = results[i][1]
        

        try:
            fig=plt.figure()
            cmap = plt.cm.jet
            N = len(set(points["label"]))
            b = len(set(points["label"]))
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
            bounds = np.linspace(0,b,b + 1)
            norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
            plt.scatter(points["x"],points["y"],c=points["label"],cmap=cmap, norm=norm, s=10)
            scoreTxt = (" Score: " + str(results[i][2])) if scored else ""
            titleTxt = "Dataset: " + str(datasetNo) + " Algorithm: " + results[i][0] + scoreTxt
            plt.title(titleTxt)
            sF = saveSubFolder + "\\" + str(results[i][0]) + ".jpg"
            #plt.show()
            plt.savefig(sF)

            plt.clf()
            plt.close()
        except:
            pass
        
        names.append(results[i][0])
        if scored:
            scores.append(results[i][2])
    if scored:
        fig=plt.figure()
        plt.bar(names, scores)
        plt.title("Scores for dataset: " + str(datasetNo))
        
        plt.ylim(ymax=1, ymin=max(0, min(scores) - 0.1))
        saveFile = datasetSaveFolder + "\\" + str(datasetNo) + ".jpg"
        #plt.show()
        plt.savefig(saveFile)

def score(algsResults, pointData):
    for i in range(0,len(algsResults)):
        algsResults[i] = algsResults[i] + (metrics.adjusted_rand_score(pointData["label"], algsResults[i][1]["label"]),)
    return algsResults

'''
useFileSaves = True
saveFolder = "Data\\Artificial\\Saves"

for currDataset in range(7,9):

    params = []
    datasetSaveFolder = saveFolder + "\\" + str(currDataset)
    pointData = arff.loadarff("Data\\Artificial\\" + str(currDataset) + ".arff")
    pointData = pd.DataFrame(pointData[0])
    resultAutoclust = []


    algsResultsDataLabels = []
    saveFolderDataLabels = datasetSaveFolder + "\\DataLabels"
    try:
        os.makedirs(saveFolderDataLabels)
    except OSError as e:
        pass

    saveFileDataLabels = saveFolderDataLabels  + "\\rawDataLabels.pkl"
 
    algsResultsDataLabels.append(("KMeans", KMeansAlg(len(set(pointData["label"])), pointData[["x","y"]])))
    algsResultsDataLabels.append(("Birch", BirchAlg(len(set(pointData["label"])), pointData[["x","y"]])))
    algsResultsDataLabels.append(("MeanShift", MeanShiftAlg(len(set(pointData["label"])), pointData[["x","y"]])))

    points = []
    for index, row in pointData.iterrows():
        points.append(P(row["x"], row["y"]))

    resultAutoclust =  auto_clust(points)
    
    
    currTrueLabel = 1
    usedLabels = []
    labelMap = []
    len([x for x in resultAutoclust])
    for i in range(0, len(resultAutoclust)):
        if len([x for x in resultAutoclust if x.label]) is 1:
            resultAutoclust[i].label = 0
        elif resultAutoclust[i].label not in labelMap:
            labelMap.append(resultAutoclust[i].label)
            resultAutoclust[i].label = currTrueLabel
            usedLabels.append(currTrueLabel )
            currTrueLabel +=1
        else:
            resultAutoclust[i].label = usedLabels[labelMap.index(resultAutoclust[i].label)]


    labelMap = usedLabels.copy()
    random.shuffle(usedLabels)
    for i in range(0, len(resultAutoclust)):
        if len([x for x in resultAutoclust if x.label]) is 1:
            continue
        else:
            prev = resultAutoclust[i].label
            ind = labelMap.index(resultAutoclust[i].label)
            resultAutoclust[i].label = usedLabels[ind]



    autoclustDF =  pd.DataFrame(columns=["x", "y", "label"])
    for point in resultAutoclust:
        autoclustDF = pd.concat([autoclustDF, pd.DataFrame([pd.Series({"x": point.x,"y": point.y,"label": point.label})])], ignore_index = True)
    resultAutoclust = ("Autoclust", autoclustDF)

    algsResultsDataLabels.append(resultAutoclust)
    algsResultsDataLabels = score(algsResultsDataLabels, pointData)
    resultAutoclust = [x for x in algsResultsDataLabels if x[0] is "Autoclust"][0]   
    saveResults(saveFolderDataLabels + "\\DataLabels.pkl", currDataset, algsResultsDataLabels, saveFolderDataLabels)         

    algsResultsNoiseLabels = []
    saveFolderNoiseLabels = datasetSaveFolder + "\\NoiseLabels"
    try:
        os.makedirs(saveFolderNoiseLabels)
    except OSError as e:
        pass

    algsResultsNoiseLabels.append(("OPTICS", OPTICSAlg(pointData[["x","y"]])))    
    algsResultsNoiseLabels.append(("DBSCAN", DBSCANAlg(pointData[["x","y"]], pointData["label"], 5)))
    algsResultsNoiseLabels = score(algsResultsNoiseLabels, pointData)
    saveResults(saveFolderNoiseLabels + "\\NoiseLabels.pkl", currDataset, algsResultsNoiseLabels, saveFolderNoiseLabels)

'''

useFileSaves = True
saveFolder = "Data\\RealWorld\\Saves"
files = ["slovenia-traffic-accidents-2016-events", "philadelphia-crime", "slovenia-illegal-dumpsites"]
for file in files:
    params = []
    datasetSaveFolder = saveFolder + "\\" + file + "\\"
    pointData = pd.read_csv("Data\\RealWorld\\" + file + ".csv")

    if file is "philadelphia-crime":
        pointData = pointData[["Lon", "Lat"]]
        pointData = pointData.drop([0, 1])
        pointData = pointData.rename(columns={"Lon": "x", "Lat": "y"})
    elif file is "slovenia-illegal-dumpsites":
        pointData = pointData[["Latitude [°]", "Longitude [°]"]]
        pointData = pointData.drop([0, 1])
        pointData = pointData.rename(columns={"Latitude [°]": "x", "Longitude [°]": "y"})
    else:
        pointData = pointData[["GeoKoordinata X", "GeoKoordinata Y"]]
        pointData = pointData.drop([0, 1])
        pointData = pointData.rename(columns={"GeoKoordinata X": "x", "GeoKoordinata Y": "y"})
    


    resultAutoclust = []

    points = []
    toRemove = []
    for index, row in pointData.iterrows():
        x = float(row.iloc[0])
        pointData["x"][index] = x
        y = float(row.iloc[1])
        pointData["y"][index] = y
        if not math.isnan(x) and not math.isnan(y):
            points.append(P(x, y))
        else:
            toRemove.append(index)
    
    for i in toRemove:
        pointData = pointData.drop(i)

    resultAutoclust =  auto_clust(points)
    

    currTrueLabel = 1
    usedLabels = []
    labelMap = []
    len([x for x in resultAutoclust])
    for i in range(0, len(resultAutoclust)):
        if len([x for x in resultAutoclust if x.label]) is 1:
            resultAutoclust[i].label = 0
        elif resultAutoclust[i].label not in labelMap:
            labelMap.append(resultAutoclust[i].label)
            resultAutoclust[i].label = currTrueLabel
            usedLabels.append(currTrueLabel )
            currTrueLabel +=1
        else:
            resultAutoclust[i].label = usedLabels[labelMap.index(resultAutoclust[i].label)]


    labelMap = usedLabels.copy()
    random.shuffle(usedLabels)
    for i in range(0, len(resultAutoclust)):
        if len([x for x in resultAutoclust if x.label]) is 1:
            continue
        else:
            prev = resultAutoclust[i].label
            ind = labelMap.index(resultAutoclust[i].label)
            resultAutoclust[i].label = usedLabels[ind]



    autoclustDF =  pd.DataFrame(columns=["x", "y", "label"])
    for point in resultAutoclust:
        autoclustDF = pd.concat([autoclustDF, pd.DataFrame([pd.Series({"x": point.x,"y": point.y,"label": point.label})])], ignore_index = True)
    resultAutoclust = ("Autoclust", autoclustDF)
 
            
        
    algsResultsAutoclustLabels = []
    
    saveFolderAutoclustLabels = datasetSaveFolder + "AutoclustLabels\\"
    try:
        os.makedirs(saveFolderAutoclustLabels)
    except OSError as e:
        pass

    saveFileAutoclustLabels = saveFolderAutoclustLabels  + "rawAutoclustLabels.pkl"

    algsResultsAutoclustLabels.append(("KMeans", KMeansAlg(len(set(autoclustDF["label"])) - 1, pointData[["x","y"]])))
    algsResultsAutoclustLabels.append(("Birch", BirchAlg(len(set(autoclustDF["label"])) - 1, pointData[["x","y"]])))
    algsResultsAutoclustLabels.append(("MeanShift", MeanShiftAlg(len(set(autoclustDF["label"])) - 1, pointData[["x","y"]])))
    algsResultsAutoclustLabels.append(("SpectralClustering", SpectralClusteringAlg(len(set(autoclustDF["label"])) - 1, pointData[["x","y"]])))
    algsResultsAutoclustLabels.append(resultAutoclust[0:2])


    algsResultsNoLabels = []
    saveFolderNoLabels = datasetSaveFolder + "NoLabels\\"
    try:
        os.makedirs(saveFolderNoLabels)
    except OSError as e:
        pass

    saveFileNoLabels = saveFolderNoLabels  + "rawNoLabels.pkl"

    algsResultsNoLabels.append(("AffinityPropagation", AffinityPropagationAlg(pointData[["x","y"]])))
    algsResultsNoLabels.append(resultAutoclust[0:2])


    algsResultsNoiseLabels = []
    saveFolderNoiseLabels = datasetSaveFolder + "NoiseLabels\\"
    try:
        os.makedirs(saveFolderNoiseLabels)
    except OSError as e:
        pass

    saveFileNoLabels = saveFolderNoLabels  + "rawNoLabels.pkl"


    algsResultsNoiseLabels.append(("OPTICS", OPTICSAlg(pointData[["x","y"]])))
    algsResultsNoiseLabels.append(resultAutoclust[0:2])


    saveResults(saveFolderAutoclustLabels + "AutoclustLabels.pkl", file, algsResultsAutoclustLabels, saveFolderAutoclustLabels, scored=False)
    saveResults(saveFolderNoiseLabels + "NoiseLabels.pkl", file, algsResultsNoiseLabels, saveFolderNoiseLabels, scored=False)
    saveResults(saveFolderNoLabels + "NoLabels.pkl", file, algsResultsNoLabels, saveFolderNoLabels, scored=False)
