import numpy as np
# from numpy.fft import fft, ifft
import cv2
import random
from skimage.morphology import skeletonize 

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
    s = 0
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
        print('iter#', s)
        s += 1
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






def img_process(image):
        
    img=cv2.resize(image,(900,300))
    cv2.imshow('img',img)
    cv2.waitKey()
    
    # mask out the cyan parts in the image 

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if (img[i,j,0]>175 & img[i,j,0] < 256) & (img[i,j,1]>175 & img[i,j,1] < 256) & (img[i,j,2]>0 & img[i,j,2] < 80):
    #             img[i,j,0] = 255
    #             img[i,j,1] = 255 
    #             img[i,j,2] = 255 

     

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('img',hsv)
    # cv2.waitKey()
    
    # # rgb mask 
    # # lower = np.uint8([175,175,0])
    # # upper = np.uint8([255,255,175])

    # # hsv mask 
    # lower = np.uint8([160,60,60])
    # upper = np.uint8([180,100,100])

    # # mask = cv2.inRange(img, lower, upper)
    # mask = cv2.inRange(hsv, lower, upper)
    # mask_inv = cv2.bitwise_not(mask)

    # cv2.imshow('img',mask)
    # cv2.waitKey()
    
    # # img = cv2.bitwise_and(img, img, mask = mask_inv)

    # hsv = cv2. bitwise_and(hsv, hsv, mask = mask_inv)
    # img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # cv2.imshow('img',img)
    # cv2.waitKey()

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

    cv2.imshow('img',binary_output)
    cv2.waitKey()


    # path_graph = []
    # print(binary_output.shape)
    # for i in range(binary_output.shape[0]):
    #     temp = []
    #     for j in range(binary_output.shape[1]):
    #         if binary_output[i,j] == 0:
    #             temp.append(1)
    #         else:
    #             temp.append(0)

    #     path_graph.append(temp)

    # path_graph = np.array(path_graph)
    # # print(path_graph)
    
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

        # eroded = cv2.erode(binary_output,element)
        # temp = cv2.dilate(eroded,element)
        # temp = cv2.subtract(binary_output,temp)
        # skel = cv2.bitwise_or(skel,temp)
        # binary_output = eroded.copy()

        # zeros = size - cv2.countNonZero(binary_output)
        # if zeros==size:
        #     done = True
    
    # binary_output = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7)))
    # cv2.imshow('img',skel)
    cv2.imshow('img',skel)
    cv2.waitKey()
    # cv2.imwrite('/home/shreyasbyndoor/path_planning_floorplan/skel_fp.jpg', skel)
    print('exported')

# img = cv2.imread("/home/shreyasbyndoor/path_planning_floorplan/edit_2.jpg")
# img_process(img)
graph_img = cv2.imread("/home/shreyasbyndoor/path_planning_floorplan/skel_fp_filled.jpg")


cv2.imshow('img',graph_img)
cv2.waitKey()

graph_img = cv2.cvtColor(graph_img, cv2.COLOR_BGR2GRAY)
path_graph = []
print(graph_img.shape)
for i in range(graph_img.shape[0]):
    temp = []
    for j in range(graph_img.shape[1]):
        if graph_img[i,j] > 230:
            temp.append(1)
        else:
            temp.append(0)

    path_graph.append(temp)

# path_graph = np.array(path_graph)

# print(path_graph.shape)

BFS(path_graph, 80, 112, 92, 98)