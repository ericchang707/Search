
# coding=utf-8
import heapq
import os
import pickle
import math
from scipy.spatial import distance



class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.
    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.
    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.
    (Hint: take a look at the module heapq)
    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.counter = 0
        
    def pop(self):
        """
        Pop top priority node from queue.
        Returns:
            The node with the highest priority.
        """
        top = self.queue[0];
        removed = heapq.heappop(self.queue)

        
        return top;
    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        appendednodes = []
        for i, val in enumerate(self.queue):
            if val[-1] == node:
                appendednodes = self.queue[:i]
                break
        for node in appendednodes:
            heapq.heappush(self.queue, node)

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
            
        """
        if len(node) == 2:
            priority = node[0] 
            task = node[1]
            heapq.heappush(self.queue, (priority, self.counter, task))
            self.counter += 1
        elif len(node) == 3:
            weight = node[0]
            priority = node[1]
            task = node[2]
            heapq.heappush(self.queue, (weight, priority, self.counter, task))
            self.counter += 1
        else:
            value = node[-1]
            heapq.heappush(self.queue, (self.counter, value))
            self.counter += 1

        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'
        Args:
            key: The key to check for in the queue.
        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.
        Args:
            other (PriorityQueue): Priority Queue to compare against.
        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.
        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.
        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]

def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    
    frontier = PriorityQueue()
    edge = {}
    path = []
    explored = set()
    
    

    explored.add(start)
    frontier.append((0, start))


    while frontier:
        if frontier.size() == 0:
            return []
        
        priority, counter, node = frontier.pop()
        
        if node == goal:
            return path

        for newnode in sorted(graph.neighbors(node)):
            if newnode not in explored:
                explored.add(newnode)
                edge[newnode] = node
                if newnode == goal:
                    path.append(goal)
                    current = goal
                    while current != start:
                        last = edge[current]
                        path.append(last)
                        current = last
                    return path[::-1]
                else:
                    frontier.append([len(newnode), newnode])
            else: 
                continue


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.
    See README.md for exercise description.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
    Returns:
        The best path as a list from the start and goal nodes (including both).
"""
    if start == goal:
        return []
    
    frontier = PriorityQueue()
    path = []
    explored = set()
    edge = {}
    path_cost = {}

    frontier.append((0, start))
    path_cost[start] = 0

    while frontier:
        if frontier.size() == 0:
            return []
        
        cost, counter, node = frontier.pop()
        if node not in explored:
            if node == goal:
                cur = goal
                while cur != start:
                    path.append(cur)
                    cur = edge[cur]
                path.append(start)
                return path[::-1]

            explored.add(node)
            for neighbor in graph.neighbors(node):
                new_cost = cost + graph.get_edge_weight(node, neighbor)
                if neighbor not in path_cost or new_cost < path_cost[neighbor]:
                    path_cost[neighbor] = new_cost
                    frontier.append((new_cost, neighbor))
                    edge[neighbor] = node
        else:
            continue
    return path



def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.
    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.
    See README.md for exercise description.
    Args:
        posi = [(x - y) ** 2 for x, y in zip(vDis, goalDis)]
    return sum(posi) ** .5
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.
    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    if v == goal:
        return 0
    vDis = graph.nodes[v]['pos']
    goalDis = graph.nodes[goal]['pos']
    return distance.euclidean(vDis, goalDis)


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    
"""

    if start == goal:
        return []

    frontier = PriorityQueue()
    frontier.append((0, start))
    initial = {}
    currentcost = {start: 0}

    while frontier:
        current = frontier.pop()[-1]
        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = currentcost[current] + graph.get_edge_weight(current, next)
            if next not in currentcost or new_cost < currentcost[next]:
                currentcost[next] = new_cost
                priority = new_cost + heuristic(graph, next, goal)
                frontier.append((priority, next))
                initial[next] = current

    # Reconstruct path
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = initial[current]
    path.append(start)
    return path[::-1]

def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    
    if start == goal:
        return []
    
    result = []
    bducs = PriorityQueue()
    bducss = PriorityQueue()
    edget = {}
    edgev = {}
    patht = {}
    pathv = {}
    resultt = []
    resultv = []
    begint = (0,start)
    beginv = (0,goal)
    bducs.append(begint)
    bducss.append(beginv)
    mol = float('inf')
    frontier = None
    patht[start] = 0
    pathv[goal] = 0
    while bducs and bducss:
        if bducs.size() == 0 or bducss.size() == 0:
            return []
        weightt, countert, nodet = bducs.pop()
        weightv, counterv, nodev = bducss.pop()
        if weightt + weightv >= mol:
            resultt.append(frontier)
            curs = frontier
            while curs != start:
                lasts = edget[curs]
                resultt.append(lasts)
                curs = lasts
            resultt = resultt[::-1]
    
            curt = frontier
            while curt != goal:
                lastg = edgev[curt]
                resultv.append(lastg)
                curt = lastg
    
            result = resultt + resultv
            return result
    
        if nodet in pathv.keys():
            if weightt + pathv[nodet] < mol:
                mol = weightt + pathv[nodet]
                frontier = nodet
    
        for newnode in graph.neighbors(nodet):
            cost = weightt + graph.get_edge_weight(nodet, newnode)
            if newnode not in patht.keys() or patht[newnode] > cost:
                patht[newnode] = cost
                bducs.append([cost, newnode])
                edget[newnode] = nodet
            else:
                continue
    
        if nodev in patht.keys():
            if weightv + patht[nodev] < mol:
                mol = weightv + patht[nodev]
                frontier = nodev
    
        for newnode in graph.neighbors(nodev):
            cost = weightv + graph.get_edge_weight(nodev, newnode)
            if newnode not in pathv.keys() or pathv[newnode] > cost:
                pathv[newnode] = cost
                bducss.append([cost, newnode])
                edgev[newnode] = nodev
            else:
                continue
    
    return result



def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.
    See README.md for exercise description.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []
    
    result = []
    bdaq = PriorityQueue()
    bdaqs = PriorityQueue()
    exploredt = set()
    exploredv = set()
    edget = {}
    edgev = {}
    patht = {}
    pathv = {}
    resultt = []
    resultv = []
    begint = [heuristic(graph, start, goal),0, start]
    beginv = [0,0,goal]
    bdaq.append(begint)
    bdaqs.append(beginv)
    mol = float('inf')
    frontier = None
    patht[start] = heuristic(graph, start, goal)
    pathv[goal] = 0
    while bdaq and bdaqs:
        if bdaq.size() == 0 or bdaqs.size() == 0:
            return []
        weightt, priorityt, countert, nodest = bdaq.pop()
        weightv, priorityv, counterv, nodesv = bdaqs.pop()
        exploredt.add(nodest)
        exploredv.add(nodesv)
        if weightt + weightv > mol:
            resultt.append(frontier)
            curt = frontier
            while curt != start:
                lastt = edget[curt]
                resultt.append(lastt)
                curt = lastt
            resultt = resultt[::-1]
    
            curv = frontier
            while curv != goal:
                lastv = edgev[curv]
                resultv.append(lastv)
                curv = lastv
    
            result = resultt + resultv
            return result
        
        heur = 0.5 * (heuristic(graph, nodest, goal) - heuristic(graph, nodest, start))

        for newnode in graph.neighbors(nodest):
            cost = weightt + graph.get_edge_weight(nodest, newnode)
            h = .5 * (heuristic(graph, newnode, goal) - heuristic(graph, newnode, start)) - heur
            f = cost + h
            if newnode not in exploredt and newnode not in bdaq:
                patht[newnode] = f
                bdaq.append([f, cost, newnode])
                edget[newnode] = nodest
                if newnode not in patht.keys():
                    patht[newnode] = f
                    bdaq.append([f, cost, newnode])
                    edget[newnode] = nodest
            elif newnode in bdaq:
                if patht[newnode] > f:
                    patht[newnode] = f
                    bdaq.remove(newnode)
                    bdaq.append([f, cost, newnode])
                    edget[newnode] = nodest
    
            if newnode in exploredv:
                if patht[newnode] + pathv[newnode] < mol:
                    mol = patht[newnode] + pathv[newnode]
                    frontier = newnode
        
        heur1 = (heuristic(graph, nodesv, start) - heuristic(graph, nodesv, goal)) * .5
        for newnode in graph.neighbors(nodesv):
            cost = weightv + graph.get_edge_weight(nodesv, newnode)
            h = 0.5 * (heuristic(graph, newnode, start) - heuristic(graph, newnode, goal)) - heur1
            f = cost + h
            if newnode not in exploredv and newnode not in bdaqs:
                pathv[newnode] = f
                bdaqs.append([f, cost, newnode])
                edgev[newnode] = nodesv
                if newnode not in pathv.keys():
                    pathv[newnode] = f
                    bdaqs.append([f, cost, newnode])
                    edgev[newnode] = nodesv
            elif newnode in bdaqs:
                if pathv[newnode] > f:
                    pathv[newnode] = f
                    bdaqs.remove(newnode)
                    bdaqs.append([f, cost, newnode])
                    edgev[newnode] = nodesv
    
            if newnode in exploredt:
                if patht[newnode] + pathv[newnode] < mol:
                    mol = patht[newnode] + pathv[newnode]
                    frontier = newnode
        
    
    
    return result

def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search
    See README.MD for exercise description.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    def makePath(re,frontier,edgea,edgeb,a,b):
        path1 = []
        path2 = []
        path1.append(frontier)
        cur1 = frontier
        while cur1 != a:
            last1 = edgea[cur1]
            path1.append(last1)
            cur1 = last1
        path1 = path1[::-1]
    
        cur2 = frontier
        while cur2 != b:
            last2 = edgeb[cur2]
            path2.append(last2)
            cur2 = last2
    
        re += path1 + path2
        
    if goals[0] == goals[1] == goals[2]:
        return []
    
    result = []
    
    triqueue1 = PriorityQueue()
    triqueue2 = PriorityQueue()
    triqueue3 = PriorityQueue()
    edge1 = {}
    edge2 = {}
    edge3 = {}
    explored1 = set()
    explored2 = set()
    explored3 = set()
    init1 = [0, goals[0]]
    init2 = [0, goals[1]]
    init3 = [0, goals[2]]
    triqueue1.append(init1)
    triqueue2.append(init2)
    triqueue3.append(init3)
    path1 = {}
    path2 = {}
    path3 = {}
    path1[goals[0]] = 0
    path2[goals[1]] = 0
    path3[goals[2]] = 0
    mol1 = float('inf')
    mol2 = float('inf')
    mol3 = float('inf')
    frontier1 = None
    frontier2 = None
    frontier3 = None
    result1 = []
    result2 = []
    result3 = []
    while triqueue1 and triqueue2 and triqueue3:
        weight1, priority1, node1 = triqueue1.top()
        weight2, priority2, node2 = triqueue2.top()
        weight3, priority3, node3 = triqueue3.top()
        if weight1 + weight2 >= mol1 and weight2 + weight3 >= mol2 and weight3 + weight1 >= mol3:
            break
    
        weight1, counter1, node1 = triqueue1.pop()
        weight2, counter2, node2 = triqueue2.pop()
        weight3, counter3, node3 = triqueue3.pop()
        explored1.add(node1)
        explored2.add(node2)
        explored3.add(node3)
    
        for newnode in sorted(graph.neighbors(node1)):
            we1 = weight1 + graph.get_edge_weight(node1, newnode)
            if newnode not in explored1 and newnode not in triqueue1:
                path1[newnode] = we1
                triqueue1.append([we1, newnode])
                edge1[newnode] = node1
            elif newnode in triqueue1:
                if path1[newnode] > we1:
                    path1[newnode] = we1
                    triqueue1.remove(newnode)
                    triqueue1.append([we1, newnode])
                    edge1[newnode] = node1
    
            if newnode in explored2:
                if path1[newnode] + path2[newnode] < mol1:
                    mol1 = path1[newnode] + path2[newnode]
                    frontier1 = newnode
    
            if newnode in explored3:
                if path1[newnode] + path3[newnode] < mol3:
                    mol3 = path1[newnode] + path3[newnode]
                    frontier3 = newnode
    
        for newnode in sorted(graph.neighbors(node2)):
            we2 = weight2 + graph.get_edge_weight(node2, newnode)
            if newnode not in explored2 and newnode not in triqueue2:
                path2[newnode] = we2
                triqueue2.append([we2, newnode])
                edge2[newnode] = node2
            elif newnode in triqueue2:
                if path2[newnode] > we2:
                    path2[newnode] = we2
                    triqueue2.remove(newnode)
                    triqueue2.append([we2, newnode])
                    edge2[newnode] = node2
    
            if newnode in explored1:
                if path1[newnode] + path2[newnode] < mol1:
                    mol1 = path1[newnode] + path2[newnode]
                    frontier1 = newnode
    
            if newnode in explored3:
                if path2[newnode] + path3[newnode] < mol2:
                    mol2 = path2[newnode] + path3[newnode]
                    frontier2 = newnode
    
        for newnode in sorted(graph.neighbors(node3)):
            we3 = weight3 + graph.get_edge_weight(node3, newnode)
            if newnode not in explored3 and newnode not in triqueue3:
                path3[newnode] = we3
                triqueue3.append([we3, newnode])
                edge3[newnode] = node3
            elif newnode in triqueue3:
                if path3[newnode] > we3:
                    path3[newnode] = we3
                    triqueue3.remove(newnode)
                    triqueue3.append([we3, newnode])
                    edge3[newnode] = node3
    
            if newnode in explored2:
                if path3[newnode] + path2[newnode] < mol2:
                    mol2 = path3[newnode] + path2[newnode]
                    frontier2 = newnode
    
            if newnode in explored1:
                if path1[newnode] + path3[newnode] < mol3:
                    mol3 = path1[newnode] + path3[newnode]
                    frontier3 = newnode
        
    
    makePath(result1, frontier1, edge1, edge2, goals[0], goals[1])
    makePath(result2, frontier2, edge2, edge3, goals[1], goals[2])
    makePath(result3, frontier3, edge3, edge1, goals[2], goals[0])
    
    if mol1 >= mol2 and mol1 >= mol3:
        result = result2[:-1] + result3
    
    if mol2 >= mol3 and mol2 >= mol1:
        result = result3[:-1] + result1
    
    if mol3 >= mol2 and mol3 >= mol1:
        result = result1[:-1] + result2
    
    return result



def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Eric Chang"





def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.
    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
