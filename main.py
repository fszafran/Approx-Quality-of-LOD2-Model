import laspy
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import time
import os 

def readLasFile(filePath):
    startTime = time.time()
    lasFile = laspy.read(filePath)
    selectedPoints = lasFile.points[lasFile.classification == 6]
    X = selectedPoints['X'] / 100
    Y = selectedPoints['Y'] / 100
    Z = selectedPoints['Z'] / 100
    coords = np.column_stack((X, Y, Z))
    globalXMin, globalXMax = X.min(), X.max()
    globalYMin, globalYMax = Y.min(), Y.max()
    endTime = time.time()
    print(f"Extracted {len(coords)} points from the las file in {endTime - startTime} seconds")
    return coords, [globalXMin, globalXMax, globalYMin, globalYMax]

def extractLOD2Points(folderPath, bbox):
    startTime = time.time()
    roofBboxDict = {}
    roofCounter = 0
    for filename in os.listdir(folderPath):
        if filename.endswith('.gml'):
            filePath = os.path.join(folderPath, filename)
            with open(filePath, 'r') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'xml')
            lowerCornerElement = soup.find('lowerCorner')
            upperCornerElement = soup.find('upperCorner')
            if lowerCornerElement is None or upperCornerElement is None:
                continue
            lowerCorner = lowerCornerElement.text.split()
            upperCorner = upperCornerElement.text.split()
            xMin = float(lowerCorner[0])
            yMin = float(lowerCorner[1])
            xMax = float(upperCorner[0])
            yMax = float(upperCorner[1])
            if xMax < bbox[0] or xMin > bbox[1] or yMax < bbox[2] or yMin > bbox[3]:
                continue
            print("Not skipped: ", filename)
            roofs = soup.find_all('RoofSurface')
            for roof in roofs:
                roofId = roof.get('gml:id')
                points = roof.find_all('pos')
                roofPoints = []
                skipRoof = False
                for point in points:
                    coords = point.text.split()
                    x = float(coords[0])
                    y = float(coords[1])
                    z = float(coords[2])
                    if x < bbox[0] or x > bbox[1] or y < bbox[2] or y > bbox[3]:
                        skipRoof = True
                        break
                    roofPoints.append([x, y, z])
                if skipRoof:
                    continue
                roofBboxDict[roofId] = roofPoints
                roofCounter += 1
    endTime = time.time()
    print(f"Extracted {roofCounter} roofs in {endTime - startTime} seconds")
    return roofBboxDict

def createPlanes(roofDict):
    startTime = time.time()
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
        planes[key] = [plane,[xMin, xMax, yMin, yMax]]
    endTime = time.time()
    print(f"Created {len(planes)} planes in {endTime - startTime} seconds")
    return planes

def getAvgErrors(planes, lasPoints):
    startTime = time.time()
    lasPoints = np.array(lasPoints)
    x = lasPoints[:,0]
    y = lasPoints[:,1]
    z = lasPoints[:,2]
    averageErrors = []
    numerator = 0
    for key in planes:
        xMin = planes[key][1][0]
        xMax = planes[key][1][1]
        yMin = planes[key][1][2]
        yMax = planes[key][1][3]
        a = planes[key][0][1]
        b = planes[key][0][2]
        c = -1
        d = planes[key][0][0]
        
        mask = (xMin <= x) & (x <= xMax) & (yMin <= y) & (y <= yMax)
        distances = (a * x[mask] + b * y[mask] + c * z[mask] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        valid_distances = distances[(distances > -1) & (distances < 1)]
        
        averageErrors.append([key, np.average(valid_distances)])
        print(f"Calculated average error for chunk {numerator}")
        numerator += 1
    endTime = time.time()
    print(f"Calculated average errors in {endTime - startTime} seconds")
    return averageErrors

def plotSurfaces(planes, averageErrors):
    error_dict = {error[0]: np.abs(error[1]) for error in averageErrors}
    min_error = min(error_dict.values())
    max_error = max(error_dict.values())
    fig, ax = plt.subplots(figsize=(10, 10))
    for key in planes:
        x = np.linspace(planes[key][1][0], planes[key][1][1], 100)
        y = np.linspace(planes[key][1][2], planes[key][1][3], 100)
        X, Y = np.meshgrid(x, y)
        Z = planes[key][1][0] + planes[key][1][1] * X + planes[key][1][2] * Y
        error = error_dict.get(key, 0)
        normalized_error = (error - min_error) / (max_error - min_error)
        cmap = plt.colormaps['YlOrRd']
        color = cmap(normalized_error)
        ax.contourf(X, Y, Z, alpha=0.5, colors=[color])
    ax.set_aspect('equal', 'box')
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0] - 0.01, ylim[1] + 0.01)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_error, vmax=max_error))
    fig.colorbar(sm, ax=ax)
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    lasFile = r'C:\Users\filo1\Desktop\szkola_sem4\Standardy_3D\pd3\pd4_lidar.laz'
    gmlFile = r'C:\Users\filo1\Desktop\szkola_sem4\Standardy_3D\pd3'
    lasPoints, globalBbox = readLasFile(lasFile)
    lod2Dict = extractLOD2Points(gmlFile, globalBbox)
    planes = createPlanes(lod2Dict)
    averageErrors= getAvgErrors(planes, lasPoints)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    plotSurfaces(planes, averageErrors)

    
    
    


