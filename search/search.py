
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
from game import Directions

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
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    new_found = []
    expanded = []
    # start state here
    startstate = problem.getStartState()
    startnode = (startstate, [])
    new_found.append(startnode)
    while len(new_found) >= 0:
        currentstate, actions = new_found[len(new_found) - 1]
        new_found.pop(-1)

        if currentstate not in expanded:
            expanded.append(currentstate)

            if problem.isGoalState(currentstate):
                return actions
            else:
                successors = problem.getSuccessors(currentstate)

                for succState, succAction, succCost in successors:
                    newaction = actions + [succAction]
                    newnode = (succState, newaction)
                    new_found.append(newnode)

def breadthFirstSearch(problem):
    q = util.Queue()

    # visitedPos holds all of the visited positions already (this is required for
    # the graph-search implementation of BFS)
    visitedPos = []

    # push starting state onto the stack with an empty path
    q.push((problem.getStartState(), []))

    # Then we can start looping, note our loop condition is if the stack is empty
    # if the stack is empty at any point we failed to find a solution
    while (not q.isEmpty()):

        # since our stack elements contain two elements
        # we have to fetch them both like this
        currentPos, currentPath = q.pop()
        # print("Currently Visiting:", currentPos, "\nPath=", end="");
        # print(currentPath);
        # then we append the currentPos to the list of visited positions
        visitedPos.append(currentPos)

        # check if current state is a goal state, if it is, return the path
        if (problem.isGoalState(currentPos)):
            return currentPath

        # obtain the list of successors from our currentPos
        successors = problem.getSuccessors(currentPos)

        # if we have successors, note that these successors have a position and the path to get there
        if (len(successors) != 0):
            # iterate through them
            for state in successors:
                # if we find one that has not already been visisted
                if ((state[0] not in visitedPos) and (state[0] not in (stateQ[0] for stateQ in q.list))):
                    # calculate the new path (currentPath + path to reach new state's position)
                    newPath = currentPath + [state[1]]
                    # push it onto the stack with the new path
                    q.push((state[0], newPath))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    q = util.PriorityQueue()

    visitedPos = []

    q.push((problem.getStartState(), []), 0)

    while (not q.isEmpty()):

        currentPos, currentPath = q.pop()

        visitedPos.append(currentPos)

        if (problem.isGoalState(currentPos)):
            return currentPath

        successors = problem.getSuccessors(currentPos)

        if (len(successors) != 0):
            for state in successors:
                if (state[0] not in visitedPos) and (state[0] not in (stateQ[2][0] for stateQ in q.heap)):
                    newPath = currentPath + [state[1]]
                    q.push((state[0], newPath), problem.getCostOfActions(newPath))

                elif (state[0] not in visitedPos) and (state[0] in (stateQ[2][0] for stateQ in q.heap)):
                    for stateQ in q.heap:
                        if stateQ[2][0] == state[0]:
                            oldPriority = problem.getCostOfActions(stateQ[2][1])

                    newPriority = problem.getCostOfActions(currentPath + [state[1]])

                    # State is cheaper with his hew father -> update and fix parent #
                    if oldPriority > newPriority:
                        newPath = currentPath + [state[1]]
                        q.update((state[0], newPath), newPriority)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

from util import PriorityQueue


class PriorityQ_and_Function(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """

    def __init__(self, problem, priorityFunc):
        "priorityFunction (item) -> priority"
        self.priorityFunc = priorityFunc  # store the priority function
        PriorityQueue.__init__(self)  # super-class initializer
        self.problem = problem

    def push(self, item, heuristic):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunc(self.problem, item, heuristic))


# Calculation of f(n) = g(n) + h(n)
def f(problem, state, heuristic):
    return problem.getCostOfActions(state[1]) + heuristic(state[0], problem)

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queueXY = PriorityQ_and_Function(problem, f)

    path = []  # Every state keeps it's path from the starting state
    visited = []  # Visited states

    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    # Add initial state. Path is an empty list #
    element = (problem.getStartState(), [])

    queueXY.push(element, heuristic)

    while (True):

        # Terminate condition: can't find solution #
        if queueXY.isEmpty():
            return []

        # Get informations of current state #
        xy, path = queueXY.pop()  # Take position and path

        # State is already been visited. A path with lower cost has previously
        # been found. Overpass this state
        if xy in visited:
            continue

        visited.append(xy)

        # Terminate condition: reach goal #
        if problem.isGoalState(xy):
            return path

        # Get successors of current state #
        succ = problem.getSuccessors(xy)

        # Add new states in queue and fix their path #
        if succ:
            for item in succ:
                if item[0] not in visited:
                    # Like previous algorithms: we should check in this point if successor
                    # is a goal state so as to follow lectures code

                    newPath = path + [item[1]]  # Fix new path
                    element = (item[0], newPath)
                    queueXY.push(element, heuristic)


class Node(object):
    def __init__(self, state=None, action=[], cost=0):
        self.state = state
        self.action = action
        self.cost = cost


directionReverseMap = {'North': 'South', 'East': 'West', 'South': 'North', 'West': 'East'}

def getReversePath(nodes):
  return [directionReverseMap[node] for node in nodes][::-1]

def biDirectionalBruteForce(problem):
    '''
    Running two BFS from start and end. Using queue for exploring nodes and dictionary for track of visited nodes
    '''
    frontierStackStart = util.Queue()
    frontierStackEnd = util.Queue()

    visitedStart = dict()
    visitedEnd = dict()

    frontierStackStart.push(problem.getStartState())  # push initial start state
    frontierStackEnd.push(problem.goal)   #push initial goal state 

    visitedStart[problem.getStartState()] = ''
    visitedEnd[problem.goal] = ''

    while not frontierStackEnd.isEmpty() and not frontierStackStart.isEmpty():     #run until all nodes explored
        #explore from front
        while not frontierStackStart.isEmpty():
            currentNode = frontierStackStart.pop()  #node to explore
            if(problem.isGoalStateForBidirectional(currentNode, visitedEnd)):
                return visitedStart[currentNode] + getReversePath(visitedEnd[currentNode])
            #exploring neighbours
            nodesNotInVisitedStart = [x for x in problem.getSuccessors(currentNode) if x[0] not in visitedStart]
            for successor in nodesNotInVisitedStart:
                frontierStackStart.push(successor[0])
                visitedStart[successor[0]] = list(visitedStart[currentNode]) + [successor[1]]  #appending next action
        
        while not frontierStackEnd.isEmpty():
            currentNodeEnd = frontierStackEnd.pop()  #node to explore
            if(problem.isGoalStateForBidirectional(currentNodeEnd, visitedStart)):
                return  getReversePath(visitedStart[currentNodeEnd]) + visitedEnd[currentNodeEnd]
            #exploring neighbours
            nodesNotInVisitedEnd = [x for x in problem.getSuccessors(currentNodeEnd) if x[0] not in visitedEnd]
            for successor in nodesNotInVisitedEnd:
                frontierStackEnd.push(successor[0])
                visitedEnd[successor[0]] = list(visitedEnd[currentNodeEnd]) + [successor[1]]  #appending next action

def biDirectionalMM(problem, heuristic):
    '''
    Meet in the middle Bidirectional search run with heuristic function
    Running two BFS from start and end along with heuristic. Using queue for exploring nodes and dictionary for track of visited nodes
    Using f(n) = g(n) + h(n) where h(n) is heuristic cost to reach destination and g(n) is actual cost from start to current state
    '''

    frontierStackStart = util.PriorityQueue()
    frontierStackEnd = util.PriorityQueue()

    visitedStart = dict()
    visitedEnd = dict()

    frontierStackStart.push((problem.getStartState()), (problem.getCostOfActions({}) + heuristic(problem.getStartState(), problem, "goalState")))  # initializing with start state and expected cost to reach goal
    frontierStackEnd.push((problem.goal), (problem.getCostOfActions({}) + heuristic(problem.goal, problem, "startState")))  # initializing with goal and expected cost to reach start state

    visitedStart[problem.getStartState()] = ''
    visitedEnd[problem.goal] = ''

    while not frontierStackStart.isEmpty() and not frontierStackEnd.isEmpty():     #run until all nodes explored with both directions at one shot

        # Run both searches at simultaneously
        currentNode = frontierStackStart.pop()

        if problem.isGoalStateForBidirectional(currentNode, visitedEnd):
            return visitedStart[currentNode] + getReversePath(visitedEnd[currentNode]) 
        
        #exploring neighbours
        nodesNotInVisitedStart = [x for x in problem.getSuccessors(currentNode) if x[0] not in visitedStart]
        for successor in nodesNotInVisitedStart:
            visitedStart[successor[0]] = list(visitedStart[currentNode]) + [successor[1]]
            frontierStackStart.push(successor[0], (problem.getCostOfActions(visitedStart[successor[0]]) + heuristic(successor[0], problem, "goalState")))

        currentNodeEnd = frontierStackEnd.pop()

        if problem.isGoalStateForBidirectional(currentNodeEnd, visitedStart):
            return visitedStart[currentNodeEnd] + getReversePath(visitedEnd[currentNodeEnd])

        #exploring neighbours
        nodesNotInVisitedEnd = [x for x in problem.getSuccessors(currentNodeEnd) if x[0] not in visitedEnd]
        for successor in nodesNotInVisitedEnd:
            visitedEnd[successor[0]] = list(visitedEnd[currentNodeEnd]) + [successor[1]]
            frontierStackEnd.push(successor[0],
                    (problem.getCostOfActions(visitedEnd[successor[0]]) + heuristic(successor[0], problem, "startState")))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bdMM0 = biDirectionalBruteForce
bdMM = biDirectionalMM
