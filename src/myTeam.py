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
        firstIndex, secondIndex, isRed, first = 'DefensiveAgent', second = 'OffensiveAgent'):
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

        # Posicio inicial de l'agent
        self.start = gameState.getInitialAgentPosition(self.index)

        # Punt mig del tauler
        self.midWidth = gameState.data.layout.width / 2

        # Totes les posicions legals del tauler on un agent pot estar
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]

        # Llista de distancies
        self.distancer.getMazeDistances()

        # Indexs dels nostres agents.
        self.team = self.getTeam(gameState)

        # Flag per saber si esta atacant.
        self.offensing = False

        # Indexs enemics.
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
                self.genericMove(enemy, gameState)
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

    def genericMove(self, enemy, gameState):
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

            # Podem descartar que una posicio sigui real
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

    def maxFunction(self, gameState, depth):
        """
        Funcio maximitzadora per obtenir el moviment (accio) més "util"
        per l'agent de l'equip.
        @params gameState
        @params depth
        @returns tuble (score, action)
        """

        # Si final
        if depth == 0 or gameState.isOver():
            return util.raiseNotDefined(), Directions.STOP

        # Moviments succesors
        actions = gameState.getLegalActions(self.index)

        # HEURISTIC millors resultats si sempre hi ha moviment requerit
        actions.remove(Directions.STOP)

        # Per cada moviment possible obtenim els succesors
        succesorStates = []
        for action in actions:
            try:
                succesorStates.append(gameState.generateSuccessor(self.index, action))
            except:
                pass

        # Obtenim els resultats dels possibles moviments enemics
        scores = []
        for successorState in succesorStates:
            scores.append(self.expectiFunction(successorState, self.enemies[0], depth)[0])

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if
                         scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return bestScore, actions[chosenIndex]

    def expectiFunction(self, gameState, enemy, depth):
        """
        This is the expectimax function from HW2. This will be called for
        each of the enemy agents. Once it goes to the next level we will use
        the max function again since we will be back on our team.
        """

        # Si final
        if depth == 0 or gameState.isOver():
            return util.raiseNotDefined(), Directions.STOP

        # Possibles moviments succesors dels enemics
        actions = gameState.getLegalActions(enemy)
        succesorStates = []
        for action in actions:
            try:
                succesorStates.append(gameState.generateSuccessor(enemy, action))
            except Exception, e:
                print e
                pass

        # If there is another ghost, then call the expecti function for the
        # next ghost, otherwise call the max function for pacman.
        if enemy < max(self.enemies):
            scores = [self.expectiFunction(successorGameState, enemy + 2, depth)[0]
                        for successorGameState in succesorStates]
        else:
            scores = [self.maxFunction(successorGameState, depth - 1)[0]
                        for successorGameState in succesorStates]

        # Millor puntuacio
        bestScore = sum(scores) / len(scores)

        return bestScore, Directions.STOP

    def enemyDistances(self, gameState):
        """
        Funcio per retornar la distacia als agenents enemics.
        En el cas que no coneguem la posicio exacte de l'enemic agafem la crence
        amb clau mes alta (més aproximada)
        @returns list amb les distacies entre la posicio de l'agent actual
        i les possibles "posicions" enemigas
        """
        dists = []
        for enemy in self.enemies:
            myPos = gameState.getAgentPosition(self.index)
            enemyPos = gameState.getAgentPosition(enemy)
            # Si no coneixem la pos enemiga
            if not enemyPos:
                enemyPos = self.beliefs[enemy].argMax()
            dists.append((enemy, self.distancer.getDistance(myPos, enemyPos)))
        return dists


class OffensiveAgent(ParentAgent):

    def registerInitialState(self, gameState):
        ParentAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)

class DefensiveAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)


    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)

        return random.choice(actions)

