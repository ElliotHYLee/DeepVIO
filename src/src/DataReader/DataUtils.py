from src.Params import *
import pandas as pd
from sys import platform
import os

def getPath(dsName='kitti', seq=0, subType='mr'):
    if dsName == 'kitti':
        path = getKITTIPath()
        path += '0'+str(seq) if seq<10 else str(seq)
        path += '/'
    return path


def getPrPath(dsName, seq, subType):
    resName = 'Results/Data/' + refBranchName() + '_' + dsName + '_'
    path = resName + subType + str(seq) #if dsName == 'airsim' else resName + str(seq)
    return path


def getImgNames(path, dsName='kitti', ts=None, subType=''):
    dsName = dsName.lower()
    imgNames = []
    temp = []
    img_dir_path = path + 'image_0/'
    files = os.listdir(img_dir_path)
    
    for i in range (0, len(files)):
        f = files[i]
        x = f.split('.')[0]
        temp.append(int(x))
    sortedIdx = sorted(range(len(temp)), key=lambda k: temp[k])
    imgNames = [img_dir_path + files[i] for i in sortedIdx]
    return imgNames

def getEnd(start, N, totalN):
    end = start+N
    if end > totalN:
        end = totalN
        N = end-start
    return end, N

class ThreadManager():
    def __init__(self, maxN=2):
        self.maxN = maxN
        self.que = []
        self.jobs = []

    def addJobs(self, t):
        self.jobs.append(t)

    def doJobs(self):
        while len(self.jobs) > 0:
            if len(self.que) < self.maxN:
                t = self.jobs.pop()
                self.que.append(t)
                t.start()

            alive_list = [job for job in self.que if job.is_alive()]
            self.que = alive_list
            del(alive_list)

