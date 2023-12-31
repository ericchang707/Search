## Overview

Search is an integral part of AI. It helps in problem solving across a wide variety of domains where a solution isn’t immediately clear.  You will implement several graph search algorithms with the goal of solving bi-directional and tri-directional search.


### The Files

| File | Description |
| ----:| :-----------|
|**__submission.py__** | Where you will implement your _PriorityQueue_, _Breadth First Search_, _Uniform Cost Search_, _A* Search_, _Bi-directional Search_, _Tri-directional Search_ |
|**_search_basic_tests.py_** | Simple unit tests to validate your searches validity and number of nodes explored. |
|**_search_submission_tests_grid.py_** | Tests searches on uniform grid and highlights path and explored nodes. |
|**_search_romania_tests.py_** | More detailed tests that run searches from a more comprehensive set of nodes on the Romania graph. |
|**_search_atlanta_tests.py_** | Tests searches on Atlanta map (takes a long time). |
|**_search_case_visualizer.py_**| Module used to visualize test cases of interest. |
|**_romania_graph.pickle_** | Serialized graph files for Romania. |
|**_romania_references.pickle** | Serialized reference cases for Romania. |
|**_explorable_graph.py_** | A wrapper around `networkx` that tracks explored nodes. **FOR DEBUGGING ONLY** |
|**_visualize_graph.py_** | Module to visualize search results. See below on how to use it. |
|**_osm2networkx.py_** | Module used by visualize graph to read OSM networks. |


## The Assignment

Your task is to implement several informed search algorithms that will calculate a driving route between two points in Romania with a minimal time and space cost.
There is a `search_basic_tests.py` file and a `search_romania_tests.py` file to help you along the way. Your searches should be executed with minimal runtime and memory overhead.

We will be using an undirected network representing a map of Romania (and an optional Atlanta graph used for the Race!).
 

#### Visualizing the Atlanta graph:

The Atlanta graph is used in some later parts of this assignment. However, it is too big to display within a Python window like Romania. As a result, when you run the bidirectional tests in **_search_atlanta_tests.py_**, it generates a JSON file in the GeoJSON format. To see the graph, you can upload it to a private GitHub Gist or use [this](http://geojson.io/) site.
If you want to see how **_visualize_graph.py_** is used, take a look at the test functions like `test_bi_ucs_atlanta_custom` in **_search_atlanta_tests.py_**






#### Warmup 1: Priority queue

_[5 points]_

In all searches that involve calculating path cost or heuristic (e.g. uniform-cost), we have to order our search frontier. It turns out the way that we do this can impact our overall search runtime.

To show this, you'll implement a priority queue which will help you in understanding its performance benefits. For large graphs, sorting all input to a priority queue is impractical. As such, the data structure you implement should have an amortized O(1) insertion and O(lg n) removal time. It should do better than the naive implementation in our tests (InsertionSortQueue), which sorts the entire list after every insertion.

In this implementation of priority queue, if two elements have the same priority, they should be served according to the order in which they were enqueued (see Hint 3).  

> **Notes**:
> 1. Please note that the algorithm runtime is not the focus of this assignment. The already-imported heapq library should achieve the desired runtime.
> 2. The local tests provided are used to test the correctness of your implementation of the Priority Queue. To verify that your implementation consistently beats the naive implementation, you might want to test it with a large number of elements.
> 3. If you use the heapq library, keep in mind that the queue will sort entries as a whole upon being enqueued, not just on the first element. This means you need to figure out a way to keep elements with the same priority in FIFO order.
> 4. You may enqueue nodes however you like, but when your Priority Queue is tested, we feed node in the form (priority, value).

#### Warmup 2: BFS

_[5 pts]_

To get you started with handling graphs, implement and test breadth-first search over the test network.

You'll complete this by writing the `breadth_first_search()` method. This returns a path of nodes from a given start node to a given end node, as a list.

For this part, it is optional to use the PriorityQueue as your frontier. You will require it from the next question onwards. You can use it here too if you want to be consistent.

> **Notes**:
> 1. You need to include start and goal in the path.
> 2. **If your start and goal are the same then just return [].**
> 3. You can access all the neighbors of a given node by calling `graph[node]`, or `graph.neighbors(node)` ONLY. 
> 4. You are not allowed to maintain a cache of the neighbors for any node. You need to use the above mentioned methods to get the neighbors.
> 5. To measure your search performance, the `explorable_graph.py` provided keeps track of which nodes you have accessed in this way (this is referred to as the set of 'Explored' nodes). To retrieve the set of nodes you've explored in this way, call `graph.explored_nodes`. If you wish to perform multiple searches on the same graph instance, call `graph.reset_search()` to clear out the current set of 'Explored' nodes. **WARNING**, these functions are intended for debugging purposes only. Calls to these functions will fail on Gradescope.
> 6. In BFS, make sure you process the neighbors in alphabetical order. Because networkx uses dictionaries, the order that it returns the neighbors is not fixed. This can cause differences in the number of explored nodes from run to run. If you sort the neighbors alphabetically before processing them, you should return the same number of explored nodes each time.
> 7. For BFS only, the autograder requires implementing an optimization trick which fully explores fewer nodes. You may find it useful to re-watch the Canvas videos for this.


#### Warmup 3: Uniform-cost search

_[10 points]_

Implement uniform-cost search, using PriorityQueue as your frontier. From now on, PriorityQueue should be your default frontier.

`uniform_cost_search()` should return the same arguments as breadth-first search: the path to the goal node (as a list of nodes).

> **Notes**:
> 1. You need to include start and goal in the path.
> 2. **If your start and goal are the same then just return [].**
> 3. The above are just to keep your results consistent with our test cases.
> 4. You can access all the neighbors of a given node by calling `graph[node]`, or `graph.neighbors(node)` ONLY. 
> 5. You can access the weight of an edge using: `graph.get_edge_weight(node_1, node_2)`. Not using this method will result in your explored nodes count being higher than it should be.
> 6. You are not allowed to maintain a cache of the neighbors for any node. You need to use the above mentioned methods to get the neighbors and corresponding weights.
> 7. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.

#### Warmup 4: A* search

_[10 points]_

Implement A* search using Euclidean distance as your heuristic. You'll need to implement `euclidean_dist_heuristic()` then pass that function to `a_star()` as the heuristic parameter. We provide `null_heuristic()` as a baseline heuristic to test against when calling a_star tests.

> **Hint**:
> You can find a node's position by calling the following to check if the key is available: `graph.nodes[n]['pos']`

> **Notes**:
> 1. You need to include start and goal in the path.
> 2. **If your start and goal are the same then just return [].**
> 3. The above are just to keep your results consistent with our test cases.
> 4. You can access all the neighbors of a given node by calling `graph[node]`, or `graph.neighbors(node)` ONLY. 
> 5. You can access the weight of an edge using: `graph.get_edge_weight(node_1, node_2)`. Not using this method will result in your explored nodes count being higher than it should be.
> 6. You are not allowed to maintain a cache of the neighbors for any node. You need to use the above mentioned methods to get the neighbors and corresponding weights.
> 7. You can access the (x, y) position of a node using: `graph.nodes[n]['pos']`. You will need this for calculating the heuristic distance.
> 8. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.

---
### Exercises
The following exercises will require you to implement several kinds of bidirectional searches. The benefits of these algorithms over uninformed or unidirectional search are more clearly seen on larger graphs. As such, during grading, we will evaluate your performance on the map of Romania included in this assignment.

For these exercises, we recommend you take a look at the resources mentioned earlier.

#### Exercise 1: Bidirectional uniform-cost search

_[20 points]_

Implement bidirectional uniform-cost search. Remember that this requires starting your search at both the start and end states.

`bidirectional_ucs()` should return the path from the start node to the goal node (as a list of nodes).

> **Notes**:
> 1. You need to include start and goal in the path. Make sure the path returned is from start to goal and not in the reverse order.
> 2. **If your start and goal are the same then just return [].**
> 3. The above are just to keep your results consistent with our test cases.
> 4. You can access all the neighbors of a given node by calling `graph[node]`, or `graph.neighbors(node)` ONLY. 
> 5. You can access the weight of an edge using: `graph.get_edge_weight(node_1, node_2)`. Not using this method will result in your explored nodes count being higher than it should be.
> 6. You are not allowed to maintain a cache of the neighbors for any node. You need to use the above mentioned methods to get the neighbors and corresponding weights.
> 7. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.

#### Exercise 2: Bidirectional A* search

_[29 points]_

Implement bidirectional A* search. Remember that you need to calculate a heuristic for both the start-to-goal search and the goal-to-start search.

To test this function, as well as using the provided tests, you can compare the path computed by bidirectional A* to bidirectional UCS search above.
`bidirectional_a_star()` should return the path from the start node to the goal node, as a list of nodes.

> **Notes**:
> 1. You need to include start and goal in the path.
> 2. **If your start and goal are the same then just return [].**
> 3. The above are just to keep your results consistent with our test cases.
> 4. You can access all the neighbors of a given node by calling `graph[node]`, or `graph.neighbors(node)` ONLY. 
> 5. You can access the weight of an edge using: `graph.get_edge_weight(node_1, node_2)`. Not using this method will result in your explored nodes count being higher than it should be.
> 6. You are not allowed to maintain a cache of the neighbors for any node. You need to use the above mentioned methods to get the neighbors and corresponding weights.
> 7. You can access the (x, y) position of a node using: `graph.nodes[n]['pos']`. You will need this for calculating the heuristic distance.
> 8. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.

#### Exercise 3: Tridirectional UCS search

_[12 points]_

Implement tridirectional search in the naive way: starting from each goal node, perform a uniform-cost search and keep
expanding until two of the three searches meet. This should be one continuous path that connects all three nodes.

For example, suppose we have goal nodes [a,b,c]. Then what we want you to do is to start at node a and expand like in a normal search. However, notice that you will be searching for both nodes b and c during this search and a similar search will start from nodes b and c. Finally, please note that this is a problem that can be accomplished without using 6 frontiers, which is why we stress that **this is not the same as 3 bi-directional searches.**

`tridirectional_search()` should return a path between all three nodes. You can return the path in any order. Eg.
(1->2->3 == 3->2->1). You may also want to look at the Tri-city search challenge question on Canvas.

> **Notes**:
> 1. You need to include start and goal in the path.
> 2. **If all three nodes are the same then just return [].**
> 3. **If there are 2 identical goals (i.e. a,b,b) then return the path [a...b] (i.e. just the path from a to b).**
> 4. The above are just to keep your results consistent with our test cases.
> 5. You can access all the neighbors of a given node by calling `graph[node]`, or `graph.neighbors(node)` ONLY. 
> 6. You can access the weight of an edge using: `graph.get_edge_weight(node_1, node_2)`. Not using this method will result in your explored nodes count being higher than it should be.
> 7. You are not allowed to maintain a cache of the neighbors for any node. You need to use the above mentioned methods to get the neighbors and corresponding weights.
> 8. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.

#### Exercise 4: Upgraded Tridirectional search

_[8 points]_

This is the heart of the assignment. Implement tridirectional search in such a way as to consistently improve on the
performance of your previous implementation. This means consistently exploring fewer nodes during your search in order
to reduce runtime. Keep in mind, we are not performing 3 bidirectional A* searches. We are searching from each of the goals towards the other two goals, in the direction that seems most promising.

The specifics are up to you, but we have a few suggestions:
 * Tridirectional A*
 * choosing landmarks and pre-computing reach values
 * ATL (A\*, landmarks, and triangle-inequality)
 * shortcuts (skipping nodes with low reach values)

`tridirectional_upgraded()` should return a path between all three nodes.

> **Notes**:
> 1. You need to include start and goal in the path.
> 2. **If all three nodes are the same then just return [].**
> 3. **If there are 2 identical goals (i.e. a,b,b) then return the path [a...b] (i.e. just the path from a to b).**
> 4. The above are just to keep your results consistent with our test cases.
> 5. You can access all the neighbors of a given node by calling `graph[node]`, or `graph.neighbors(node)` ONLY. 
> 6. You can access the weight of an edge using: `graph.get_edge_weight(node_1, node_2)`. Not using this method will result in your explored nodes count being higher than it should be.
> 7. You are not allowed to maintain a cache of the neighbors for any node. You need to use the above mentioned methods to get the neighbors and corresponding weights.
> 8. You can access the (x, y) position of a node using: `graph.nodes[n]['pos']`. You will need this for calculating the heuristic distance.
> 9. We will provide some margin of error in grading the size of your 'Explored' set, but it should be close to the results provided by our reference implementation.
     
     
#### Final Task: Return your name
_[1 point]_

A simple task to wind down the assignment. Return your name from the function aptly called `return_your_name()`.

