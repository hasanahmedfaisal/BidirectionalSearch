
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

    if(problem.isGoalState(problem.getStartState())):
        return []
    initialState=problem.getStartState()
    node = None
    frontierStack = util.Stack()
    frontierStack.push(Node(initialState))
    closedSet = set()
    failure = 0
    while 5>0:
        if(frontierStack.isEmpty()):
            failure = 1 
        node = frontierStack.pop()
        nodeState = node.state
        nodeActions = node.action
        if(problem.isGoalState(nodeState)):
            break
        if nodeState not in closedSet:
            closedSet.add(nodeState)
            childNodes = problem.getSuccessors(nodeState)

            for childNode in childNodes:
                fullActions = [*nodeActions,childNode[1]]
                tempNode=Node(childNode[0],fullActions)
                frontierStack.push(tempNode)
         
    if failure == 1:
        print("failure")
        return []
    
    return nodeActions
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    if(problem.isGoalState(problem.getStartState())):
        return []
    initialState=problem.getStartState()
    node = None
    frontierStack = util.Queue()
    frontierStack.push(Node(initialState))
    closedSet = set()
    failure = 0
    while 5>0:
        if(frontierStack.isEmpty()):
            failure = 1 
        node = frontierStack.pop()
        nodeState = node.state
        nodeActions = node.action
        if(problem.isGoalState(nodeState)):
            break
        if nodeState not in closedSet:
            closedSet.add(nodeState)
            childNodes = problem.getSuccessors(nodeState)

            for childNode in childNodes:
                fullActions = [*nodeActions,childNode[1]]
                tempNode=Node(childNode[0],fullActions)
                frontierStack.push(tempNode)
         
    if failure == 1:
        print("failure")
        return []
    
    return nodeActions
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    if(problem.isGoalState(problem.getStartState())):
        return []
    initialState=problem.getStartState()
    node = None
    frontierStack = util.PriorityQueue()
    frontierStack.push(Node(initialState),0)
    closedSet = set()
    failure = 0
    while 5>0:
        if(frontierStack.isEmpty()):
            failure = 1 
        node = frontierStack.pop()
        nodeState = node.state
        nodeActions = node.action
        nodeCosts = node.cost
        if(problem.isGoalState(nodeState)):
            break
        if nodeState not in closedSet:
            closedSet.add(nodeState)
            childNodes = problem.getSuccessors(nodeState)

            for childNode in childNodes:
                fullActions = [*nodeActions,childNode[1]]
                fullCosts = nodeCosts+childNode[2]
                tempNode=Node(childNode[0],fullActions,fullCosts)
                frontierStack.push(tempNode,fullCosts)
         
    if failure == 1:
        print("failure")
        return []
    
    return nodeActions
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    if(problem.isGoalState(problem.getStartState())):
        return []
    initialState=problem.getStartState()
    node = None
    frontierStack = util.PriorityQueue()
    frontierStack.push(Node(initialState),0)
    closedSet = set()
    failure = 0
    while True:
        if(frontierStack.isEmpty()):
            failure = 1 
        node = frontierStack.pop()
        nodeState = node.state
        nodeActions = node.action
        nodeCosts = node.cost
        if(problem.isGoalState(nodeState)):
            break
        if nodeState not in closedSet:
            closedSet.add(nodeState)
            childNodes = problem.getSuccessors(nodeState)

            for childNode in childNodes:
                fullActions = [*nodeActions,childNode[1]]
                fullCosts = nodeCosts+childNode[2]
                tempNode=Node(childNode[0],fullActions,fullCosts)
                itemCost = fullCosts + heuristic(childNode[0],problem) #applying heuristic
                frontierStack.push(tempNode,itemCost)
         
    if failure == 1:
        print("failure")
        return []
    
    return nodeActions
    util.raiseNotDefined()


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
            for successor in problem.getSuccessors(currentNode):
                if(successor[0] in visitedStart):
                    continue
                frontierStackStart.push(successor[0])
                visitedStart[successor[0]] = list(visitedStart[currentNode]) + [successor[1]]  #appending next action
        
        while not frontierStackEnd.isEmpty():
            currentNodeEnd = frontierStackEnd.pop()  #node to explore
            if(problem.isGoalStateForBidirectional(currentNodeEnd, visitedStart)):
                return  getReversePath(visitedStart[currentNodeEnd]) + visitedEnd[currentNodeEnd]
            #exploring neighbours
            for successor in problem.getSuccessors(currentNodeEnd):
                if(successor[0] in visitedEnd):
                    continue
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

        for state in problem.getSuccessors(currentNode): 
            if state[0] in visitedStart:
                continue
            visitedStart[state[0]] = list(visitedStart[currentNode]) + [state[1]]
            frontierStackStart.push(state[0], (problem.getCostOfActions(visitedStart[state[0]]) + heuristic(state[0], problem, "goalState")))

        currentNodeEnd = frontierStackEnd.pop()

        if problem.isGoalStateForBidirectional(currentNodeEnd, visitedStart):
            return visitedStart[currentNodeEnd] + getReversePath(visitedEnd[currentNodeEnd])

        #exploring neighbours

        for state in problem.getSuccessors(currentNodeEnd):
            if state[0] in visitedEnd:
                continue
            visitedEnd[state[0]] = list(visitedEnd[currentNodeEnd]) + [state[1]]
            frontierStackEnd.push(state[0],
                    (problem.getCostOfActions(visitedEnd[state[0]]) + heuristic(state[0], problem, "startState")))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bdMM0 = biDirectionalBruteForce
bdMM = biDirectionalMM
