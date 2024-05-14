import laspy
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readLasFile(filePath, bbox):
    lasFile = laspy.read(filePath)
    selectedPoints = lasFile.points[lasFile.classification == 6]
    coords = []
    with open('building_point_cloud.txt', 'w') as file:
        for point in selectedPoints:
            if bbox[0] <= point.X/100 <= bbox[1] and bbox[2] <= point.Y/100 <= bbox[3]:
                file.write(f"{point.X/100} {point.Y/100} {point.Z/100}\n")
                coords.append([point.X/100, point.Y/100, point.Z/100])
    return coords

def extractLOD2Points(filePath):
    with open(filePath, 'r') as f:
        content = f.read()
    soup = BeautifulSoup(content, 'xml')
    roofs = soup.find_all('RoofSurface')
    roofBboxDict = {}
    globalXMin = np.inf
    globalXMax = -np.inf
    globalYMin = np.inf
    globalYMax = -np.inf
    for roof in roofs:
        roofId = roof.get('gml:id')
        points = roof.find_all('pos')
        if roofId not in roofBboxDict:
                roofBboxDict[roofId] = []
        for point in points:
            coords = point.text.split()
            x = float(coords[0])
            y = float(coords[1])
            z = float(coords[2])
            if x < globalXMin:
                globalXMin = x
            if x > globalXMax:
                globalXMax = x
            if y < globalYMin:
                globalYMin = y
            if y > globalYMax:
                globalYMax = y
            roofBboxDict[roofId].append([x, y, z])
    return roofBboxDict, [globalXMin, globalXMax, globalYMin, globalYMax]

def createPlanes(roofDict):
    planes = {}
    for key in roofDict:
        xList = []
        yList = []
        zList = []
        xMin = np.inf
        xMax = -np.inf
        yMin = np.inf
        yMax = -np.inf
        for i in range(len(roofDict[key])):
            xList.append(roofDict[key][i][0])
            yList.append(roofDict[key][i][1])
            zList.append(roofDict[key][i][2])
        x = np.array(xList)
        y = np.array(yList)
        z = np.array(zList)
        xMin = min(x)
        xMax = max(x)
        yMin = min(y)
        yMax = max(y)
        A = np.array((np.ones_like(x),x,y)).T
        B = z
        plane = np.linalg.inv(A.T @ A) @ A.T @ B
        planes[key] = [key, plane,[xMin, xMax, yMin, yMax]]
    return planes

def getChunksOfLasPoints(lasPoints, planes):
    lasPoints = np.array(lasPoints)
    x = lasPoints[:,0]
    y = lasPoints[:,1]
    z = lasPoints[:,2]
    chunks = {}
    for key in planes:
        xMin = planes[key][2][0]
        xMax = planes[key][2][1]
        yMin = planes[key][2][2]
        yMax = planes[key][2][3]
        chunk = []
        for i in range(len(x)):
            if xMin <= x[i] <= xMax and yMin <= y[i] <= yMax:
                chunk.append([x[i], y[i], z[i]])
        chunks[key] = chunk
    return chunks

def getAvgErrors(planes, chunks):
    averageErrors=[]
    finalPoints = []
    for key in planes:
        c = -1
        b = planes[key][1][2]
        a = planes[key][1][1]
        d = planes[key][1][0]
        distances =[]
        for i in range(len(chunks[key])):
            distance = (a * chunks[key][i][0] + b * chunks[key][i][1] + c * chunks[key][i][2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
            if distance < 1 and distance > -1:
                distances.append(distance)
                finalPoints.append([chunks[key][i][0], chunks[key][i][1], chunks[key][i][2]])
        averageErrors.append([key,np.average(distances)])
    return averageErrors, finalPoints

def plotCartogram(averagesErrors):
    ids = [error[0][-2:] if error[0][-2:].isdigit() else error[0][-1] for error in averagesErrors]
    errors = [error[1] for error in averagesErrors]
    plt.bar(ids, errors)
    plt.xlabel('Roof ID')
    plt.ylabel('Average Error [m]')
    plt.title('Average Error for each roof')
    plt.size = (20, 10)
    plt.show()


if __name__ == '__main__':
    lasFile = r'C:\Users\filo1\Desktop\szkola_sem4\Standardy_3D\pd3\lidar.laz'
    gmlFile = r'C:\Users\filo1\Desktop\szkola_sem4\Standardy_3D\pd3\lod_2_jeden_budynek.gml'
    lod2Dict, globalBbox = extractLOD2Points(gmlFile)
    lasPoints = readLasFile(lasFile, globalBbox)
    # #create a plane from lasPoints using least squares method...
    # plane = createPlanes('building_point_cloud.txt')
    # finalPoints = errors('building_point_cloud.txt', plane)[0]
    # averageError = errors('building_point_cloud.txt', plane)[1]
    # distances = errors('building_point_cloud.txt', plane)[2]
    chunks = getChunksOfLasPoints(lasPoints, createPlanes(lod2Dict))
    averageErrors = getAvgErrors(createPlanes(lod2Dict), chunks)[0]
    plotCartogram(averageErrors)
    # # write final points to a file
    # with open('final_point.txt', 'w') as file:
    #     for point in finalPoints:
    #         file.write(f"{point[0]} {point[1]} {point[2]}\n")
    # plot_plane_and_points(plane, np.array(finalPoints),distances)
    
    
    


