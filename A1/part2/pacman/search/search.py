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


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
        """
    
    from util import Stack
    print("\n problem:  ", problem,"  ENd \n")
    DFS_stack=Stack()

    visited = [] 
    path = []
    if problem.isGoalState(problem.getStartState()):
        return []

    DFS_stack.push((problem.getStartState(),[]))

    while(True):

        if DFS_stack.isEmpty():
            return []
        xy,path = DFS_stack.pop()
        visited.append(xy)

        if problem.isGoalState(xy):
            return path

        succ = problem.getSuccessors(xy)

        if succ:
            for child in succ:
                if child[0] not in visited:

                    update_path = path + [child[1]]
                    DFS_stack.push((child[0],update_path))     


# ________________________________________________________________

class _RecursiveDepthFirstSearch(object):
    '''
        => Output of 'recursive' dfs should match that of 'iterative' dfs you implemented
        above. 

        Key Point: Remember in tutorial you were asked to expand the left-most child 
        first for dfs and bfs for consistency. If you expanded the right-most
        first, dfs/bfs would be correct in principle but may not return the same
        path. 

        => Useful Hint: self.problem.getSuccessors(node) will return children of 
        a node in a certain "sequence", say (A->B->C), If your 'recursive' dfs traversal 
        is different from 'iterative' traversal, try reversing the sequence.  

    '''
    def __init__(self, problem):
        " Do not change this. " 
        # You'll save the actions that recursive dfs found in self.actions. 
        self.actions = [] 
        # Use self.explored to keep track of explored nodes.  
        self.explored = set()
        self.problem = problem

    def RecursiveDepthFirstSearchHelper(self, node):
        '''
        args: start node 
        outputs: bool => True if path found else Fasle.
        '''
        if node in self.explored:
           return False

        if(len(node)==2):
            node=(node,)
        
        if self.problem.isGoalState(node[0]):
            self.actions.append(node[1])
            return True
        
        self.explored.add(node[0])

        for successor in reversed(self.problem.getSuccessors(node[0])):
            if successor[0] in self.explored:
                continue
            possible=self.RecursiveDepthFirstSearchHelper(successor)
            if possible:
                if len(node)>2:
                    self.actions.append(node[1])
                return True
        return False





def RecursiveDepthFirstSearch(problem):
    " You need not change this function. "
    # All your code should be in member function 'RecursiveDepthFirstSearchHelper' of 
    # class '_RecursiveDepthFirstSearch'."
    print("\nProblem : ",problem,"  End\n")

    node = problem.getStartState() 
    print("\nnode : ",node,"  End\n")
    rdfs = _RecursiveDepthFirstSearch(problem)
    print("\nrdfs : ",rdfs,"  End\n")
    path_found = rdfs.RecursiveDepthFirstSearchHelper(node)
    return list(reversed(rdfs.actions)) # Actions your recursive calls return are in opposite order.
# ________________________________________________________________


def depthLimitedSearch(problem, limit = 210):

    """
    Search the deepest nodes in the search tree first as long as the
    nodes are not not deeper than 'limit'.

    For medium maze, pacman should find food for limit less than 130. 
    If your solution needs 'limit' more than 130, it's bogus.
    Specifically, for:
    'python pacman.py -l mediumMaze -p SearchAgent -a fn=dls', and limit=130
    pacman should work normally.  

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    Autograder cannot test this function.  

    Hints: You may need to store additional information in your frontier(queue).

        """

    "Start of Your Code"
    # pass
    "End of Your Code"
    from util import Stack
    print("\n problem:  ", problem,"  ENd \n")
    DLS_stack=Stack()

    visited = []    
    path = [] 
    print("\n Problem: ",problem,"  \n")
    DLS_stack.push(((problem.getStartState(),0),[]))

    while(True):

        if DLS_stack.isEmpty():
            return []

        xy,path = DLS_stack.pop() 
        if(xy[1]<=limit):
            if problem.isGoalState(xy[0]):
                return path
        
            visited.append(xy[0])

            succ = problem.getSuccessors(xy[0])
        
            if succ:
                for child in succ:
                    if child[0] not in visited:

                        update_path = path + [child[1]]
                        DLS_stack.push(((child[0],xy[1]+1),update_path)) 
        
        
          


# ________________________________________________________________

class _RecursiveDepthLimitedSearch(object):
    '''
        => Output of 'recursive' dfs should match that of 'iterative' dfs you implemented
        above. 
        Key Point: Remember in tutorial you were asked to expand the left-most child 
        first for dfs and bfs for consistency. If you expanded the right-most
        first, dfs/bfs would be correct in principle but may not return the same
        path. 

        => Useful Hint: self.problem.getSuccessors(node) will return children of 
        a node in a certain "sequence", say (A->B->C), If your 'recursive' dfs traversal 
        is different from 'iterative' traversal, try reversing the sequence.  

    '''
    def __init__(self, problem):
        " Do not change this. " 
        # You'll save the actions that recursive dfs found in self.actions. 
        self.actions = [] 
        # Use self.explored to keep track of explored nodes.  
        self.explored = set()
        self.problem = problem
        self.current_depth = 0
        self.depth_limit = 210 # For medium maze, You should find solution for depth_limit not more than 204.

    def RecursiveDepthLimitedSearchHelper(self, node):
        '''
        args: start node 
        outputs: bool => True if path found else Fasle.
        '''

        "Start of Your Code"
        # pass
        "End of Your Code"
        if(self.current_depth>self.depth_limit):
            return False
        
        if node in self.explored:
               return False

        if(len(node)==2):
            node=(node,)
        
        if self.problem.isGoalState(node[0]):
            self.actions.append(node[1])
            return True
        
        self.explored.add(node[0])

        for successor in self.problem.getSuccessors(node[0])[::-1]:
            if successor[0] in self.explored:
                continue
            self.current_depth+=1
            possible=self.RecursiveDepthLimitedSearchHelper(successor)
            self.current_depth-=1

            if possible:
                if len(node)>2:
                    self.actions.append(node[1])
                return True
        
        return False


def RecursiveDepthLimitedSearch(problem):
    "You need not change this function. All your code in member function RecursiveDepthLimitedSearchHelper"
    node = problem.getStartState() 
    rdfs = _RecursiveDepthLimitedSearch(problem)
    path_found = rdfs.RecursiveDepthLimitedSearchHelper(node)
    return list(reversed(rdfs.actions)) # Actions your recursive calls return are in opposite order.
# ________________________________________________________________


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    from util import Queue

    BFS_queue = Queue()

    visited = [] 
    path = [] 

    if problem.isGoalState(problem.getStartState()):
        return []

    BFS_queue.push((problem.getStartState(),[]))

    while(True):

        if BFS_queue.isEmpty():
            return []

        xy,path = BFS_queue.pop() 
        visited.append(xy)


        if problem.isGoalState(xy):
            return path

        succ = problem.getSuccessors(xy)

        if succ:
            for child in succ:
                if child[0] not in visited and child[0] not in (state[0] for state in BFS_queue.list):
                    update_path = path + [child[1]] 
                    BFS_queue.push((child[0],update_path))


def uniformCostSearch(problem):
    """Search the node of least total cost first.
       You may need to pay close attention to util.py.
       Useful Reminder: Note that problem.getSuccessors(node) returns "step_cost". 

       Key Point: If a node is already present in the queue with higher path cost, 
       you'll update its cost. (Similar to pseudocode in figure 3.14 of your textbook.). Be careful, 
       autograder cannot catch this bug. 
    """

    "Start of Your Code"
    # pass
    "End of Your Code"

    from util import PriorityQueue

    UCS_queue = PriorityQueue()

    visited = [] 
    path = [] 

    if problem.isGoalState(problem.getStartState()):
        return []

    UCS_queue.push((problem.getStartState(),[]),0)

    while(True):

        if UCS_queue.isEmpty():
            return []

        ((xy,path),c) = UCS_queue.pop() 
        visited.append(xy)
        check=xy,path

        if problem.isGoalState(xy):
            return path

        succ = problem.getSuccessors(xy)
        if succ:
            for child in succ:
                if child[0] not in visited and (child[0] not in (data[2][0] for data in UCS_queue.heap)):


                    newPath = path + [child[1]]
                    pri = problem.getCostOfActions(newPath)

                    UCS_queue.push((child[0],newPath),pri)
                elif child[0] not in visited and (child[0] in (data[2][0] for data in UCS_queue.heap)):
                    i=0
                    for data in UCS_queue.heap:
                        if data[2][0] == child[0]:
                            i-=1
                            oldprice = problem.getCostOfActions(data[2][1])

                    newprice = problem.getCostOfActions(path + [child[1]])

                    if oldprice > newprice:
                        newPath = path + [child[1]]
                        UCS_queue.Update_priority((child[0],newPath),newprice,i,UCS_queue.count)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
from util import PriorityQueue
class MyPriorityQueueWithFunction(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """
    def  __init__(self, problem, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction      # store the priority function
        PriorityQueue.__init__(self)        # super-class initializer
        self.problem = problem
    def push(self, item, heuristic):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(self.problem,item,heuristic))
def f(problem,state,heuristic):
    
    return problem.getCostOfActions(state[1]) + heuristic(state[0],problem)
def aStarSearch(problem, heuristic=nullHeuristic):
    '''
    Pay clos attention to util.py- specifically, args you pass to member functions. 

    Key Point: If a node is already present in the queue with higher path cost, 
    you'll update its cost (Similar to pseudocode in figure 3.14 of your textbook.). Be careful, 
    autograder cannot catch this bug.

    '''
    "Start of Your Code"
    # pass
    "End of Your Code"
    from util import PriorityQueue

    A_queue = PriorityQueue()

    visited = [] 
    path = [] 

    
    if problem.isGoalState(problem.getStartState()):
        return []

    element = (problem.getStartState(),[])
    cost=problem.getCostOfActions(element[1]) + heuristic(element[0],problem)
    A_queue.push(element,cost)

    while(True):
        if A_queue.isEmpty():
            return []

        ((xy,path),c) = A_queue.pop() 
        visited.append(xy)
        check=xy,path

        if problem.isGoalState(xy):
            return path

        succ = problem.getSuccessors(xy)
        if succ:
            for child in succ:
                if child[0] not in visited and (child[0] not in (data[2][0] for data in A_queue.heap)):

                    newPath = path + [child[1]]
                    element = (child[0],newPath)
                    cost=problem.getCostOfActions(element[1]) + heuristic(element[0],problem)
                    A_queue.push(element,cost)

                elif child[0] not in visited and (child[0] in (data[2][0] for data in A_queue.heap)):
                    i=0
                    for data in A_queue.heap:
                        if data[2][0] == child[0]:
                            i-=1
                            oldprice = problem.getCostOfActions(data[2][1])

                    newprice = problem.getCostOfActions(path + [child[1]])

                    if oldprice > newprice:
                        newPath = path + [child[1]]
                        element = (child[0],newPath)
                        cost=problem.getCostOfActions(element[1]) + heuristic(element[0],problem)
                        A_queue.Update_priority((child[0],newPath),cost,i,A_queue.count)
    
    
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
rdfs = RecursiveDepthFirstSearch
dls = depthLimitedSearch
rdls = RecursiveDepthLimitedSearch
astar = aStarSearch
ucs = uniformCostSearch
