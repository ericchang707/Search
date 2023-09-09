# coding=utf-8

# Author: Raymond Jia
# Authored: 8/23/2023
# Last Updated: 8/25/2023 by Raymond Jia

import pickle
import unittest

import networkx

from explorable_graph import ExplorableGraph
from submission import a_star, bidirectional_a_star, \
  bidirectional_ucs, breadth_first_search, compute_landmarks, \
  euclidean_dist_heuristic, null_heuristic, tridirectional_search, \
  tridirectional_upgraded, uniform_cost_search

class SearchRomaniaTests(unittest.TestCase):
  """
  Almost comprehensive testing for all search algorithms in this assignment using the Romania map.

  Passing all of these local test scenarios should almost directly translate to fully passing on Gradescope.

  The map provided on Gradescope is not identical to the Romania graph, however
  the test cases provided here are based on the locally provided Romania graph 
  and should be almost exhaustive of possible pathing scenarios that may occur on Gradescope.

  How to use this file?

  This file should be placed in your Assignment 1 repo, at the same level as your submission.py
  file. This file also requires the pickle files 'romania_graph.pickle' and 'romania_references.pickle'
  to be present in the repo.

  To run the entire file (not recommended unless you are almost done) navigate to the repo
  in your Anaconda prompt and run 'python search_romania_tests.py'. If you pass all the tests, you
  should see an 'OK' after a few seconds.

  To run individual tests (recommended) navigate to the repo in your Anaconda prompt and run
  'python search_romania_tests.py SearchRomaniaTests.TheTestName'. For example if you want to
  run the BFS test, your command should be 'python search_romania_tests.py SearchRomaniaTests.test_bfs_romania'.

  All tests will generate txt files with logging information about how your submission fared. This is
  to help you identify what specific cases on the Romania map you are failing and why, and you are expected
  to manually debug and identify what is wrong with the cases you are failing before you go to Office Hours
  for help. You can then take the specific cases you are failing and visualize them using
  the provided 'search_case_visualizer.py' file.

  This file does not test your Priority Queue implementation. It is expected that your Priority Queue implementation
  is working appropriately. This file also does not test runtime, it is expected that your implementation
  is relatively optimized and does not timeout on Gradescope.

  You may use this file as an inspiration to design and write your own test files.
  """

  def setUp(self):
    """Setup Romania graph data"""

    with(open("romania_graph.pickle", "rb")) as romania_file:
        romania = pickle.load(romania_file)
    self.romania = ExplorableGraph(romania)
    self.romania.reset_search()

    with(open("romania_references.pickle", "rb")) as romania_ref_file:
        self.romania_refs = pickle.load(romania_ref_file)

  @staticmethod
  def get_path_cost(graph, path):
    """
    Calculate the total cost of a path by summing edge weights

    Args:
      graph (ExplorableGraph): Graph that contains the path
      path (list(nodes)): List of nodes from src to dst

    Returns:
      Sum of edge weights in path
    """

    pairs = zip(path, path[1:])
    return sum([graph.get_edge_data(a, b)['weight'] for a, b in pairs])
  
  @staticmethod
  def is_continuous_path(graph, path):
    """
    Checks if the provided path is a continuous path

    Args:
      graph (ExplorableGraph): Undirected graph with path
      path (list(nodes)): List of nodes

    Returns:
      Boolean True if path is continuous, False otherwise
    """

    for i in range(0, len(path)-1):
        edges = networkx.edges(graph, path[i])
        if not any([path[i+1] == v for e,v in edges]):
            return False
    return True
  
  def is_valid_path(self, graph, src_node, dst_node, path):
    """
    Checks if the provided path (two goals) is a valid path

    Args:
      graph (ExplorableGraph): Undirected graph with path
      src_node (node): Key for the start node
      dst_node (node): Key for the end node
      path (list(nodes)): List of nodes from src to dst

    Returns:
      Boolean True if path is valid, False otherwise
    """

    # Check if stationary path
    if src_node == dst_node:
      return not path
    
    # Check endpoints
    if not ((path[0] == src_node and path[-1] == dst_node) or (path[-1] == src_node and path[0] == dst_node)):
      return False
    
    # Check path continuity
    return self.is_continuous_path(graph, path)
  
  def is_valid_path_tridir(self, graph, goals, path):
    """
    Checks if the provided path (three goals) is a valid path

    Args:
      graph (ExplorableGraph): Undirected graph with path
      goals (list(nodes)): Key for the goal nodes
      path (list(nodes)): List of nodes from src to dst

    Returns:
      Boolean True if path is valid, False otherwise
    """

    # Check endpoints
    if not (path[0] in goals and path[-1] in goals):
      return False
    
    # Check path contains all goals
    for goal in goals:
      if goal not in path:
        return False
    
    # Check path continuity
    return self.is_continuous_path(graph, path)
  
  def is_optimal_path(self, test_name, case, path):
    """
    Check if path provided is shortest path matching references

    Args:
      test_name: Name of test
      case (tuple(node)): Goal nodes
      path (list(node)): Student's solution path
    
    Returns:
      Boolean, Path Cost, Expected Cost
    """
    
    path_cost = len(path) if case == "test_bfs_romania" else self.get_path_cost(self.romania, path)
    ref_path = self.romania_refs[test_name][case][0]
    # Check if ref_path is accidentally flipped
    if len(path) > 0 and ref_path[-1] == path[0]:
      ref_path = ref_path[::-1]
    ref_path_cost = len(ref_path) if case == "test_bfs_romania" else self.get_path_cost(self.romania, ref_path)

    matches_ref = path == ref_path
    return matches_ref, ref_path, path_cost, ref_path_cost
  
  def is_allowed_explored_cnt(self, test_name, case, explored):
    """
    Check if explored is within allowed bounds of reference

    Args: 
      test_name: Name of test
      case (tuple(node)): Goal nodes
      path (dict(node: int)): Student's explored

    Returns:
      Boolean, Explored Count, Expected Explored Count Max
    """

    explored_cnt = sum(explored.values())
    ref_explored_cnt_max = self.romania_refs[test_name][case][1]

    is_within_bounds = explored_cnt <= ref_explored_cnt_max
    return is_within_bounds, explored_cnt, ref_explored_cnt_max
  
  def log_messages(self, test_name, msg_log):
    with(open("local_test-" + test_name + ".txt", 'w')) as outFile:
      outFile.writelines('\n'.join(msg_log))
  
  def run_romania_test(self, test_name, test_func, goal_count, **kwargs):
    """General testing structure"""

    # Message log collection
    msg_log = []

    # Collect all paths
    student_solns = {}
    for goal_case in self.romania_refs[test_name]:
      self.romania.reset_search()
      if (goal_count == 2):
        student_solns[goal_case] = (test_func(self.romania, goal_case[0], goal_case[1], **kwargs), self.romania.explored_nodes())
      elif (goal_count == 3):
        student_solns[goal_case] = (test_func(self.romania, list(goal_case), **kwargs), self.romania.explored_nodes())
      else:
        self.assertTrue(False, msg="Test setup incorrectly")
        return
    
    # Check if all paths are valid
    failed = 0
    for case in student_solns:
      with self.subTest(case = case):
        if (goal_count == 2):
          verdict = self.is_valid_path(self.romania, case[0], case[1], student_solns[case][0])
          err = "Path %s for start node '%s' to goal node '%s' is not a valid path" % (student_solns[case][0], case[0], case[1])
        elif (goal_count == 3):
          verdict = self.is_valid_path_tridir(self.romania, case, student_solns[case][0])
          err = "Path %s for goal nodes %s is not a valid path" % (student_solns[case][0], case)
        if not verdict:
          msg_log.append(err)
          failed += 1
        self.assertTrue(verdict, msg=err)
    if failed == 0:
      msg_log.append(test_name + " - Valid paths tests passed.")
    else:
      msg_log.append(test_name + " - " + str(failed) + " tests failed.")
      self.log_messages(test_name, msg_log)
      return
        
    # Check if all paths match optimal
    failed = 0
    for case in student_solns:
      with self.subTest(case = case):
        verdict, expected_path, path_cost, expected_path_cost = self.is_optimal_path(test_name, case, student_solns[case][0])
        err = "Path %s for goal nodes %s does not match reference. Path cost was %s and expected path cost was %s. Expected path is %s" % (student_solns[case][0], case, path_cost, expected_path_cost, expected_path)
        if not verdict:
          msg_log.append(err)
          failed += 1
        self.assertTrue(verdict, msg=err)
    if failed == 0:
      msg_log.append(test_name + " - Optimal paths tests passed.")
    else:
      msg_log.append(test_name + " - " + str(failed) + " tests failed.")
      self.log_messages(test_name, msg_log)
      return
    
    # Check if explored nodes is within allowed amount
    failed = 0
    for case in student_solns:
      with self.subTest(case = case):
        verdict, explored_cnt, expected_explored_cnt_max = self.is_allowed_explored_cnt(test_name, case, student_solns[case][1])
        err = "Path %s for goal nodes %s explored more nodes than allowed maximum. Explored count was %s and max allowed count was %s" % (student_solns[case][0], case, explored_cnt, expected_explored_cnt_max)
        if not verdict:
          msg_log.append(err)
          failed += 1
        self.assertTrue(verdict, msg=err)
    if failed == 0:
      msg_log.append(test_name + " - Explored nodes tests passed.")
    else:
      msg_log.append(test_name + " - " + str(failed) + " tests failed.")
      self.log_messages(test_name, msg_log)
      return
    
    msg_log.append("All " + test_name + " passed.")
    self.log_messages(test_name, msg_log)
  
  def test_bfs_romania(self):
    """Test breadth first search with Romania data"""

    self.run_romania_test("test_bfs_romania", breadth_first_search, 2)

  def test_ucs_romania(self):
    """Test uniform cost search with Romania data"""

    self.run_romania_test("test_ucs_romania", uniform_cost_search, 2)

  def test_a_star_null_romania(self):
    """Test A* search with Romania data and Null heuristic"""

    self.run_romania_test("test_a_star_null_romania", a_star, 2, heuristic=null_heuristic)

  def test_a_star_euclidean_romania(self):
    """Test A* search with Romania data and the Euclidean heuristic."""

    self.run_romania_test("test_a_star_euclidean_romania", a_star, 2, heuristic=euclidean_dist_heuristic)

  def test_bi_ucs_romania(self):
    """Test Bi-uniform cost search with Romania data."""

    self.run_romania_test("test_bi_ucs_romania", bidirectional_ucs, 2)

  def test_bi_a_star_null_romania(self):
    """Test Bi-A* search with Romania data and the Null heuristic."""

    self.run_romania_test("test_bi_a_star_null_romania", bidirectional_a_star, 2, heuristic=null_heuristic)

  def test_bi_a_star_euclidean_romania(self):
    """Test Bi-A* search with Romania data and the Euclidean heuristic."""

    self.run_romania_test("test_bi_a_star_euclidean_romania", bidirectional_a_star, 2, heuristic=euclidean_dist_heuristic)

  def test_tri_ucs_romania(self):
    """Test Tri-UC search with Romania data."""

    self.run_romania_test("test_tri_ucs_romania", tridirectional_search, 3)

  def test_tri_upgraded_romania(self):
    """Test upgraded tri search with Romania data and the Euclidean heuristic."""

    landmarks = compute_landmarks(self.romania)
    self.run_romania_test("test_tri_upgraded_romania", tridirectional_upgraded, 3, heuristic=euclidean_dist_heuristic, landmarks=landmarks)


if __name__ == '__main__':
  unittest.main()