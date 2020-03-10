import numpy as np
import cv2
import random
from skimage.morphology import skeletonize 

M = 300
N = 900

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
    prev = []
    for c in range(M):
        temp1 = [] 
        temp2 = []
        for d in range(N):
            temp1.append(False)
            temp2.append([])
        visited.append(temp1)
        prev.append(temp2)
    # print(prev)
    
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
        # print(node)

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
                prev[i + row[k]][ j + col[k]].append((i,j))
            
            
    

    if (min_dist != np.inf):
        print("The shortest path from source to destination has length ",min_dist) 
    else:
        print("Destination can't be reached from given source")
    
    path = []

    m = x
    n = y
    while prev[m][n] != []:
        path.append(prev[m][n][0])
        tup = prev[m][n][0]
        m = tup[0]
        n = tup[1]
        

    print(path)
    # print(len(path))
    return path



def img_process(image):
        
    # img=cv2.resize(image,(900,300))
    # cv2.imshow('img',img)
    # cv2.waitKey()
    # img = cv2.imwrite("/home/shreyasbyndoor/Indoor_localization_CSL/color_fp_resize.jpg", img)
    
    """
    Apply sobel edge detection on input image in x, y direction
    """
    # 1. Convert the image to gray scale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    cv2.imshow('img',binary_output)
    cv2.waitKey()

    
    size = np.size(binary_output)
    skel = np.zeros(binary_output.shape,np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    done = False 
    # Skeletonize the image 
    while True: 
        #Step 2: Open the image
        open_img = cv2.morphologyEx(binary_output, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(binary_output, open_img)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(binary_output, element)
        skel = cv2.bitwise_or(skel,temp)
        binary_output = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(binary_output)==0:
            break

    
    cv2.imshow('img',skel)
    cv2.waitKey()
    cv2.imwrite('/home/shreyasbyndoor/Indoor_localization_CSL/skel_fp.jpg', skel)
    print('exported')

# img = cv2.imread("/home/shreyasbyndoor/Indoor_localization_CSL/color_fp_resize.jpg")
# img_process(img)

graph_img = cv2.imread("/home/shreyasbyndoor/Indoor_localization_CSL/skel_fp_filled_2.jpg")


cv2.imshow('img',graph_img)
cv2.waitKey()

graph_img = cv2.cvtColor(graph_img, cv2.COLOR_BGR2GRAY)
path_graph = []
print(graph_img.shape)
for i in range(graph_img.shape[0]):
    temp = []
    for j in range(graph_img.shape[1]):
        if graph_img[i,j] > 10:
            temp.append(1)
        else:
            temp.append(0)

    path_graph.append(temp)

path_graph = np.array(path_graph)

# path_coordinates = BFS(path_graph, 82, 130, 223, 770)  ## room 223 to 257 works
path_coordinates = BFS(path_graph, 223, 127,80,730)  ## room 214 to 246
# path_coordinates = BFS(path_graph, 223, 327, 81, 293)    ## room 205 to 229
# path_coordinates = BFS(path_graph,223,110,164,810) # 215 to 276A

# print(path_coordinates)

# // DRAW PATH ON FLOORPLAN 
img = cv2.imread("/home/shreyasbyndoor/Indoor_localization_CSL/color_fp_resize.jpg")
# print(img.shape)

vertchange = 1  #corresponds to change in x coordinates 
horizchange = 1 #corresponds to change in y coordinates 

# to help detect change in path 
xprev = path_coordinates[0][0]
yprev = path_coordinates[0][1]

for i in path_coordinates:
    x = i[0]
    y = i[1]

    img[x,y,0] = 0
    img[x,y,1] = 255
    img[x,y,2] = 0


for i in range(len(path_coordinates)):
    point = path_coordinates[i]
    x = point[0]
    y = point[1]

    if (horizchange == 1):
        if abs(x-xprev) > 5:
            vertchange = 1
            horizchange = 0
            print("turning point at ",xprev,",",yprev)
            xprev = x 
            yprev = y

    if (vertchange == 1):
        if abs(y-yprev) > 5:
            vertchange = 0
            horizchange = 1
            print("turning point at ",xprev,",",yprev)
            xprev = x 
            yprev = y
    

cv2.imshow('img',img)
cv2.waitKey()