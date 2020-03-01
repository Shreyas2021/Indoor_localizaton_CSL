import numpy as np


M = 10 
N = 10


# Below arrays details all 4 possible movements from a cell
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
    print(prev)

    #initially all cells are unvisited

    #create an empty queue
    q = []

    #mark source cell as visited and enqueue the source node
    visited[i][j] = True
    q.append([ i, j, 0 ])

    #stores length of longest path from source to destination
    min_dist = np.inf 

    s = 0
     
    #run till queue is not empty
    while (q != []):
        #pop front node from queue and process it
        node = q.pop(0)
        print(node)

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
    

#Shortest path in a Maze

#input maze
mat = [[1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 1]]
    
print(len(mat))
#Find shortest path from source(0, 0) to
#destination(7, 5)
BFS(mat, 0, 0, 7, 5)
# mat = np.array(mat)
# indices = [[0,0], [0,1]]
# print(mat[indices[0]])
