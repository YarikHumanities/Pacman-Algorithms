# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
"""
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    from util import Stack

   
    
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    queue = Queue()
    visited = []
    queue.push((problem.getStartState(), []))

    while not queue.isEmpty():
        curr_state, path_to_curr = queue.pop()
        if curr_state not in visited:
            visited.append(curr_state)
            if problem.isGoalState(curr_state):
                return path_to_curr
            for successor, direction, cost in problem.getSuccessors(curr_state):
                temp_path = path_to_curr+[direction]
                queue.push((successor, temp_path))

    #util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    priority_queue = PriorityQueue()
    visited = []
    priority_queue.push((problem.getStartState(), [], 0), 0)

    while not priority_queue.isEmpty():
        curr_state, path_to_curr, cost = priority_queue.pop()
        print("Curr_state, path_to_curr, cost: ", curr_state," ", path_to_curr," ", cost)
        if curr_state not in visited:
            visited.append(curr_state)

            if problem.isGoalState(curr_state):
                return path_to_curr
            
            for successor, successor_direction, successor_cost in problem.getSuccessors(curr_state):
                print("Successor of ", curr_state, " - ", successor, " with direction ", successor_direction, " and cost ", successor_cost)

                temp_path = path_to_curr + [successor_direction]
                print("Temp_path: ", temp_path)

                temp_cost = cost + successor_cost
                print("Temp_cost: ", temp_cost)

                heuristic_value = temp_cost + heuristic(successor, problem)
                print("Heuristic value: ", heuristic_value)
                priority_queue.push((successor, temp_path, temp_cost), heuristic_value)

    util.raiseNotDefined()

def greedySearch(problem: SearchProblem, heuristic=nullHeuristic):

    from util import PriorityQueue
    priority_queue = PriorityQueue()
    visited = []
    priority_queue.push((problem.getStartState(), []), 0)

    while not priority_queue.isEmpty():
        curr_state, path_to_curr = priority_queue.pop()
        if curr_state not in visited:
            visited.append(curr_state)

            if problem.isGoalState(curr_state):
                return path_to_curr
            
            for successor, successor_direction, successor_cost in problem.getSuccessors(curr_state):
                temp_path = path_to_curr + [successor_direction]
                priority_queue.push((successor, temp_path), heuristic(successor, problem))

    util.raiseNotDefined()
# Abbreviations
bfs = breadthFirstSearch #Lee Algorithm
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
greedy = greedySearch
