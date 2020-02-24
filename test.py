import numpy as np
# from numpy.fft import fft, ifft
import cv2
import random


M = 300
N = 300

row = [- 1, 0, 0, 1]
col = [0, - 1, 1, 0]
#Function to check if it is possible to go to position(row, col)
#from current position.The function returns false if (row, col)
#is not a valid position or has value 0 or it is already visited
def isValid(mat, visited, row, col):
    return ((row >= 0) and (row < M) and (col >= 0) and (col < N) and (mat[row][col]==1) and (visited[row][col]==0))
 

#Find Shortest Possible Route in a matrix mat from source
#cell(i, j) to destination cell(x, y)
def BFS(mat, i, j, x, y):
#construct a matrix to keep track of visited cells

    visited = []

    for c in range(M):
        temp = [] 
        for d in range(N):
            temp.append(False)
        visited.append(temp)

    #initially all cells are unvisited

    #create an empty queue
    q = []

    #mark source cell as visited and enqueue the source node
    visited[i][j] = True
    q.append([ i, j, 0 ])

    #stores length of longest path from source to destination
    min_dist = np.inf 

    #run till queue is not empty
    while (q != []):
        #pop front node from queue and process it
        node = q.pop(0)
        print(node)
        # q.pop()

        #(i, j) represents current cell and dist stores its
        #minimum distance from the source
        i = node[0] 
        j = node[1]
        dist = node[2]

        #if destination is found, update min_dist and stop
        if (i == x and j == y):
            min_dist = dist
            break
        #check for all 4 possible movements from current cell
        #and enqueue each valid movement

        for k in range(4):
            #check if it is possible to go to position
            #(i + row[k], j + col[k]) from current position
            if (isValid(mat, visited, i + row[k], j + col[k])):
                #mark next cell as visited and enqueue it
                visited[i + row[k]][j + col[k]] = True
                q.append([ i + row[k], j + col[k], dist + 1 ])

    if (min_dist != np.inf):
        print("The shortest path from source to destination has length ",min_dist) 
    else:
        print("Destination can't be reached from given source")




img = cv2.imread('B148-2.png')
img=cv2.resize(img,(1000,1000))
# cv2.imshow('img',img)
# cv2.waitKey()
# lane = lanenet_detector()
# lane.gradient_thresh(img)
"""
Apply sobel edge detection on input image in x, y direction
"""
# 1. Convert the image to gray scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Gaussian blur the image
img = cv2.GaussianBlur(img, (3, 3), 0)

# 3. Use cv2.Sobel() to find derievatives for both X and Y Axis
grad_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
grad_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)

# 4. Use cv2.addWeighted() to combine the results
grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

grad = cv2.convertScaleAbs(grad)
# Apply threshold
binary_output = cv2.threshold(grad, 25, 100,
                                cv2.THRESH_BINARY)[1]
                    
for i in range(binary_output.shape[0]):
    for j in range(binary_output.shape[1]):
        if binary_output[i,j] == 100:
            binary_output[i,j] = 0
        else:
            binary_output[i,j] = 255



path_graph = []
print(binary_output.shape)
for i in range(binary_output.shape[0]):
    temp = []
    for j in range(binary_output.shape[1]):
        if binary_output[i,j] == 0:
            temp.append(1)
        else:
            temp.append(0)

    path_graph.append(temp)


print(path_graph)

cv2.imshow('img',binary_output)
cv2.waitKey()

# BFS(path_graph, 0, 0,  )