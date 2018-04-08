# -*- coding: utf-8 -*-
# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(
        firstIndex, secondIndex, isRed, first = 'PolsAgent', second = 'AmatsAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

constDepth = 2
move_range = [-1, 0, 1]

class ParentAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        # Get the starting position of our agent.
        self.start = gameState.getInitialAgentPosition(self.index)

        # Get the midpoint of the board.
        self.midWidth = gameState.data.layout.width / 2

        # Get the legal positions that agents could be in.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]

        # So we can use maze distance.
        self.distancer.getMazeDistances()

        # Get our team agent indexes.
        self.team = self.getTeam(gameState)

        # Flag for offense.
        self.offensing = False

        # Get our enemy indexes.
        self.enemies = self.getOpponents(gameState)

        # Initialize the belief to be 1 at the initial position for each of the
        # opposition agents. The beliefs will be a dictionary of dictionaries.
        # The inner dictionaries will hold the beliefs for each agent.
        self.beliefs = {}
        for enemy in self.enemies:
          self.beliefs[enemy] = util.Counter()
          self.beliefs[enemy][gameState.getInitialAgentPosition(enemy)] = 1.

    def chooseAction(self, gameState):
        """
        Parent action selection method.
        This method update agent's map beliefs.
        @param: gameState string map of game.
        @returns str action with selected move direction ex. North
        """

        # Distances to listened sounds (list of 4 integers with distances to sounds)
        noisyDistances = gameState.getAgentDistances()

        # Makes a copy of game state
        newState = gameState.deepCopy()

        # For any enemy agent tries to get visual contact of enemy
        for enemy in self.enemies:
            # None if no visual contact else enemy position tuple
            enemyPos = gameState.getAgentPosition(enemy)
            if enemyPos:
                new_belief = util.Counter()
                new_belief[enemyPos] = 1.0
                self.beliefs[enemy] = new_belief
            else:
                # If not visual contact observe and move
                self.elapseTime(enemy, gameState)
                self.observe(enemy, noisyDistances, gameState)

        # Using the most probable position update the game state.
        # In order to use expectimax we need to be able to have a set
        # position where the enemy is starting out.
        for enemy in self.enemies:
            prob_pos = self.beliefs[enemy].argMax()
            conf = game.Configuration(prob_pos, Directions.STOP)
            newState.data.agentStates[enemy] = game.AgentState(conf, newState.isRed(prob_pos) != newState.isOnRedTeam(
              enemy))

        # TODO imp effi
        action = self.maxFunction(newState, depth=constDepth)[1]

        return action

    def elapseTime(self, enemy, gameState):
        """
        Comprova totes les possibles posicions succesores i que sigui legal el moviment i
        es reaparteix de manera uniforme la distribucio de probabilitats i en retorna una
        """
        new_belief = util.Counter()
        # legalPositions is a list of tuples (x,y)
        for oldPos in self.legalPositions:
            # Get the new probability distribution.
            newPosDist = util.Counter()

            # Mirem les possibles posicions succesores
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if not (abs(i) == 1 and abs(j) == 1):
                        pos_pos = (oldPos[0] + i, oldPos[1] + j)
                        if pos_pos in self.legalPositions:
                            newPosDist[pos_pos] = 1.0

            # Normalize to be unifom assuming the movement is random.
            newPosDist.normalize()

            # Get the new belief distibution.
            for newPos, prob in newPosDist.items():
                # Update the probabilities for each of the positions.
                new_belief[newPos] += prob * self.beliefs[enemy][oldPos]

        # Normalize and update the belief.
        new_belief.normalize()
        self.beliefs[enemy] = new_belief

    def observe(self, enemy, observation, gameState):
        """
        Funció d'acotament per determinar més exactament la possible
        posició enemiga (creences).
        @param: gameState string map of game.
        @param: observation int list of 4 elements with noisy distance
                between current agent and all agents.
        """
        # Get the noisy observation for the current enemy.
        noisyDistance = observation[enemy]

        # Get the position of the calling agent.
        current_pos = gameState.getAgentPosition(self.index)

        # Create new dictionary to hold the new beliefs for the current enemy.
        new_belief = util.Counter()

        # Actualitzem les creences de les posicions legalse del tauler.
        for p in self.legalPositions:
            # Distancia real entre l'agent actual i la posicio iteració
            trueDistance = util.manhattanDistance(current_pos, p)

            # Probabilitat tenint en compte la distancia real i la probable
            # P(e_t|x_t).
            emissionModel = gameState.getDistanceProb(trueDistance, noisyDistance)

            # Pode, descartar que una posicio sigui real
            # comrovant el tipus d'agent que es pacman o fasntasma i sapiguent
            # a quin camp es troba.
            if self.red:
                pac = p[0] < self.midWidth
            else:
                pac = p[0] > self.midWidth

            # Si la distancia real es inferior a 6 la descartem perque tindria
            # visio de l'objectiu i no estaria al vector de distancies de sons
            # si no a les distancies reals
            if trueDistance <= 5:
                new_belief[p] = 0.
            elif pac != gameState.getAgentState(enemy).isPacman:
                new_belief[p] = 0.
            else:
                # P(x_t|e_1:t) = P(x_t|e_1:t) * P(e_t:x_t).
                new_belief[p] = self.beliefs[enemy][p] * emissionModel

        # Si no tenim creences inicialitzem de manera uniforme per cada posicio
        # altrament normalitzem i actualitzem amb les noves creences
        if new_belief.totalCount() == 0:
            self.initializeBeliefs(enemy)
        else:
            new_belief.normalize()
            self.beliefs[enemy] = new_belief

    def initializeBeliefs(self, enemy):
        """
        Inicialitza les creencies de manera uniforme per totes les possibles posicions
        i en normalitza les probabilitats (que la suma total sigui 1)
        """
        self.beliefs[enemy] = util.Counter()
        for pos in self.legalPositions:
            # This value of 1, could be anything since we will normalize it.
            self.beliefs[enemy][pos] = 1.0

        self.beliefs[enemy].normalize()


class PolsAgent(ParentAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

class AmatsAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

