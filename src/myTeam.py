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
import random, util
from game import Directions
import game
import sys

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

  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

EXPECTIMAX_DEPTH = 2
MOVE_RANGE = [-1, 0, 1]


class ParentAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # Punt mig del tauler
        self.midWidth = gameState.data.layout.width / 2

        # Totes les posicions legals del tauler on un agent pot estar
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]

        # Llista de distancies
        self.distancer.getMazeDistances()

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

        # Distàncies als sons escoltats. Tenim una llista de 4 enters
        # que representen les distàncies
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
                self.actualitzarCreences(enemy)
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
        action = self.maxFunction(newState, depth=EXPECTIMAX_DEPTH)[1]

        return action

    def actualitzarCreences(self, enemy):
        """
        Comprova totes les possibles posicions succesores i que sigui legal el moviment i
        es reparteix de manera uniforme la distribucio de probabilitats
        """
        newBelief = util.Counter()
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
                newBelief[newPos] += prob * self.beliefs[enemy][oldPos]

        # Normalize and update the belief.
        newBelief.normalize()
        self.beliefs[enemy] = newBelief

    def observe(self, enemy, observation, gameState):
        """
        Funció d'acotament per determinar més exactament la possible
        posició enemiga (creences).
        @param: gameState state of the game.
        @param: observation int list of 4 elements with noisy distance
                between current agent and all agents.
        """
        # Obtenim la distància al enemic
        noisyDistance = observation[enemy]

        # La nostra posició
        myPos = gameState.getAgentPosition(self.index)

        # Diccionari per a guardar les suposicions per al enemic actual
        newBelief = util.Counter()

        # Actualitzem les creences de les posicions legalse del tauler.
        for pos in self.legalPositions:
            # Distancia real entre l'agent actual i la posicio iteració
            trueDistance = util.manhattanDistance(myPos, pos)

            # Probabilitat tenint en compte la distancia real i la probable
            # P(e_t|x_t).
            emissionModel = gameState.getDistanceProb(trueDistance, noisyDistance)

            # Podem descartar que una posicio sigui real
            # comrovant el tipus d'agent que es pacman o fasntasma i sapiguent
            # a quin camp es troba.
            if self.red:
                pac = pos[0] < self.midWidth
            else:
                pac = pos[0] > self.midWidth

            # Si la distancia real es inferior a 6 la descartem perque tindria
            # visio de l'objectiu i no estaria al vector de distancies de sons
            # si no a les distancies reals
            if trueDistance <= 5:
                newBelief[pos] = 0.
            elif pac != gameState.getAgentState(enemy).isPacman:
                newBelief[pos] = 0.
            else:
                # P(x_t|e_1:t) = P(x_t|e_1:t) * P(e_t:x_t).
                newBelief[pos] = self.beliefs[enemy][pos] * emissionModel

        # Si no tenim creences inicialitzem de manera uniforme per cada posicio
        # altrament normalitzem i actualitzem amb les noves creences
        if newBelief.totalCount() == 0:
            self.initializeBeliefs(enemy)
        else:
            newBelief.normalize()
            self.beliefs[enemy] = newBelief

    def initializeBeliefs(self, enemy):
        """
        Inicialitza les creències de manera uniforme per totes les possibles posicions
        i en normalitza les probabilitats (que la suma total sigui 1)
        """
        self.beliefs[enemy] = util.Counter()
        for pos in self.legalPositions:
            # Assignem un valor de probabilitat per defecte
            self.beliefs[enemy][pos] = 1.0

        self.beliefs[enemy].normalize()

    def evaluateGameState(self, gameState):
        """
        Evalua l'estat del joc
        """
        util.raiseNotDefined()

    def maxFunction(self, gameState, depth):
        """
        Funcio maximitzadora per obtenir el moviment (acció) més "útil"
        per l'agent de l'equip.

        @params gameState
        @params depth
        @returns tuple (score, action)
        """

        # Si final
        if depth == 0 or gameState.isOver():
            return self.evaluateGameState(gameState), Directions.STOP

        # Moviments succesors
        actions = gameState.getLegalActions(self.index)

        # HEURISTIC millors resultats si sempre hi ha moviment requerit
        actions.remove(Directions.STOP)

        # Per cada moviment possible obtenim els succesors
        succesorStates = []
        for action in actions:
            succesorStates.append(gameState.generateSuccessor(self.index, action))

        # Obtenim els resultats dels possibles moviments enemics
        scores = []
        for successorState in succesorStates:
            scores.append(self.expectiFunction(successorState, self.enemies[0], depth)[0])

        bestScore = max(scores)
        bestIndices = [
            index for index in xrange(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)

        return bestScore, actions[chosenIndex]

    def expectiFunction(self, gameState, enemy, depth):
        """
        Funció expectiMax. Serà cridada per a cada un dels agents enemics.
        """

        # Si final
        if depth == 0 or gameState.isOver():
            return self.evaluateGameState(gameState), Directions.STOP

        # Possibles moviments succesors dels enemics
        actions = gameState.getLegalActions(enemy)
        succesorStates = []
        for action in actions:
            succesorStates.append(gameState.generateSuccessor(enemy, action))

        # Si hi ha algun fantasma, cridem la expectiFunction per a l'altre
        # fantasma, sino anem al seguent nivell de l'expectimax cridant
        # la maxFunction
        if enemy < max(self.enemies):
            scores = [
                self.expectiFunction(successorGameState, enemy + 2, depth)[0]
                for successorGameState in succesorStates
            ]
        else:
            scores = [
                self.maxFunction(successorGameState, depth - 1)[0]
                for successorGameState in succesorStates
            ]

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
        self.retreating = False

    def chooseAction(self, gameState):
        enoughFood = 7

        # Si tenim prou menjar és moment de retirar-nos a un lloc segur
        if gameState.getAgentState(self.index).numCarrying > enoughFood:
            self.retreating = True
        else:
            self.retreating = False
        return ParentAgent.chooseAction(self, gameState)

    def evaluateGameState(self, gameState):
        """
        Mètode heurístic que evalua l'estat del tauler basant-se en el punt de
        vista de un agent ofensiu. Valorarà:
            - La quantitat i la distància dels fantasmes enemics
            - La quantitat i la distància dels powerups enemics
            -
        :return:
        """
        # Obtenim la nostra posicio
        myPos = gameState.getAgentPosition(self.index)

        # Obtenir distancies cap als dos enemics
        enemyDists = []
        nearestGhostDistance = sys.maxsize
        for enemy in self.enemies:
            # if not gameState.getAgentState(enemy).isPacman:
            enemyPos = gameState.getAgentPosition(enemy)
            if enemyPos:
                # Si trobem algun enemic, volem trobar la distancia
                distance = self.distancer.getDistance(myPos, enemyPos)
                enemyDists.append(distance)
                if distance < nearestGhostDistance:
                    # En el cas que sigui el fantasma mes proper,
                    # si està espantat, interessa atacar-lo
                    scaredTimer = gameState.getAgentState(enemy).scaredTimer
                    nearestGhostDistance = distance
                    if 0 < scaredTimer and distance < 4:
                        enemyDists[-1] *= -100

        # Obtenir la puntuació segons si hi ha enemics a prop
        if enemyDists:
            minDists = min(enemyDists)
            enemiesDistanceScore = minDists if minDists < 6 else 0
        else:
            enemiesDistanceScore = 0

        # Si tenim prou menjar hem de fugir a deixar-lo a lloc segur
        if self.retreating:
            # Obtenir distancia al mig del tauler
            tableHeight = gameState.data.layout.height
            distanceToMiddle = min(
                [
                    self.distancer.getDistance(
                        myPos, (self.midWidth, i)
                    )
                    for i in xrange(tableHeight)
                    if (self.midWidth, i) in self.legalPositions
                ]
            )
            return -1000 * distanceToMiddle + enemiesDistanceScore

        # Obtenir distancia cap al powerup
        if self.red:
            powerups = gameState.getBlueCapsules()
        else:
            powerups = gameState.getRedCapsules()

        powerupsDistances = tuple(
            self.distancer.getDistance(myPos, pwup) for pwup in powerups
        )

        if powerupsDistances:
            powerupsDistancesScore = min(powerupsDistances) * 2
        else:
            powerupsDistancesScore = 0

        # Menjar proper per anar a buscar-lo
        targetFood = self.getFood(gameState).asList()
        nearestFoodDistance = min([
            self.distancer.getDistance(myPos, food)
            for food in targetFood
        ])

        return (
            2 * self.getScore(gameState) - 500 * len(targetFood) -
            3 * nearestFoodDistance - 1000 * len(powerups) -
            5 * powerupsDistancesScore + 100 * enemiesDistanceScore
        )


class DefensiveAgent(ParentAgent):
    '''
    L'agent defensor és l'encarregat de que quan detecti que un agent enemic
    esta en el nostre territori, s'encarregui de persaguir-lo retornant la
    accio que mes s'apropi a la posible posició de l'enemic.
    En cas que no hi hagi cap agent enemic en el nostre tarreny pasara a mode
    ofensiu executant la funcio d'evaluacio de l'agent ofeensiu.
    '''
    def registerInitialState(self, gameState):
        # Posicio actual de l'agent
        ParentAgent.registerInitialState(self, gameState)
        # Variable per saber si l'agent ha de pasar a mode ofensiu
        self.offensing = False

    def chooseAction(self, gameState):
        # Mirem s'hi ha algun enemic en forma de pacman (atacant)
        invaders = [enemy_agent
                    for enemy_agent in self.enemies
                    if gameState.getAgentState(enemy_agent).isPacman]

        # Mirem si tenim el power-up actiu.
        powerTimes = [gameState.getAgentState(enemy).scaredTimer
                      for enemy in self.enemies]

        # Si no hi ha enemics atacant o tenim el power-up actiu pasem l'agent
        # a l'atac
        self.offensing = not invaders or min(powerTimes) > 8

        return ParentAgent.chooseAction(self, gameState)

    def evaluateGameState(self, gameState):
        # Posicio del nostre agent actual
        currentPos = gameState.getAgentPosition(self.index)

        # Distancies fins als agents enemics
        enemyDistances = self.enemyDistances(gameState)

        # Mirem s'hi ha algun enemic en forma de pacman (atacant)
        invaders = [enemy_agent
                    for enemy_agent in self.enemies
                    if gameState.getAgentState(enemy_agent).isPacman]

        minimPacDistance = False
        minimGhostDistance = False
        # Calculem la distancia més propera al pacman enemic més proper si n'hi ha
        # Calculem la distancia minima al fantasma més proper si n'hi ha
        for id, dist in enemyDistances:
            minimPacDistance = dist if (not minimPacDistance or dist < minimPacDistance) and id in invaders else minimPacDistance
            minimGhostDistance = dist if (not minimGhostDistance or dist < minimGhostDistance) and id not in invaders else minimGhostDistance
        minimPacDistance = 0 if not minimPacDistance else minimPacDistance
        minimGhostDistance = 0 if not minimGhostDistance else minimGhostDistance

        # Calculem la distancia minima al menjar
        minFoodDistance = False
        foodList = self.getFood(gameState).asList()
        for food in foodList:
            dist = self.distancer.getDistance(currentPos, food)
            minFoodDistance = dist if not minFoodDistance or dist < minFoodDistance else minFoodDistance
        minFoodDistance = 0 if not minFoodDistance else minFoodDistance

        # Calcul de la distancia minima al power-up
        powUps = self.getCapsulesYouAreDefending(gameState)
        minPowerDistance = False
        for power in powUps:
            dist = self.getMazeDistance(currentPos, power)
            minPowerDistance = dist if not minPowerDistance or dist < minPowerDistance else minPowerDistance
        minPowerDistance = 0 if not minPowerDistance else minPowerDistance

        if not self.offensing:
            # Si estem defensant ens fixem en el nombre d'enemics i la distancia minima entre ells i l'agent actual
            return -999999 * len(invaders) - 10 * minimPacDistance - minPowerDistance
        else:
            return 2 * self.getScore(gameState) - 100 * len(foodList) - 3 * minFoodDistance + minimGhostDistance


