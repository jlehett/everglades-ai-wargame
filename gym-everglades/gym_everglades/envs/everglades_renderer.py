from gym.envs.classic_control import rendering
import time
import random
import math

# Some adjustable constants for rendering
CONTROL_POINT_SIZE = 30
CONTROL_POINT_Y_SPACING = 150
UNIT_SIZE = 7
RANDOM_NOISE = 25
IN_TRANSIT_SPACE = 20
TIME_BETWEEN_FRAMES = 0.0
SCREEN_DIM = (900, 500)

class EvergladesRenderer:
    def __init__(self, game):
        self.game = game
        self.screen_width, self.screen_height = SCREEN_DIM
        self.viewer = None

    def getLatestData(self):
        self.player1State = self.game.player_state(0)
        self.player1Board = self.game.board_state(0)
        self.player2State = self.game.player_state(1)
        self.player2Board = self.game.board_state(1)
    
    def getNodeData(self, nodeNum, playerNum):
        boardList = [self.player1Board, self.player2Board]
        board = boardList[playerNum]
        nodeData = board[4*nodeNum:4*nodeNum+4]
        return nodeData

    def createNodesArray(self):
        # Create player 1's node representations
        self.p1NodeRenderInfo = []
        for nodeNum in range(1, 12):
            node = rendering.make_circle(CONTROL_POINT_SIZE)
            translate = rendering.Transform()
            node.add_attr(translate)
            self.viewer.add_geom(node)
            self.p1NodeRenderInfo.append({
                'node': node,
                'translate': translate
            })

    def createNodeConnections(self):
        connections = [
            (0,1), (0,3), (1,2), (1,4), (2,3), (2,4), (2,5), (2,6), (3,6),
            (4,7), (4,8), (5,8), (6,8), (6,9), (7,8), (7,10), (8,9), (9,10)
        ]
        for connection in connections:
            connectionGeom = rendering.make_polyline([
                self.p1NodeRenderInfo[connection[0]]['translate'].translation,
                self.p1NodeRenderInfo[connection[1]]['translate'].translation
            ])
            connectionGeom.set_color(0, 0, 0)
            self.viewer.add_geom(connectionGeom)

    def createUnitGroups(self):
        # Create player 1's unit groups
        self.p1UnitRenderInfo = []
        self.p2UnitRenderInfo = []
        for playerNum, playerState in enumerate([self.player1State, self.player2State]):
            for groupNum in range(12):
                # Grab the unit group info
                groupInfo = playerState[1+groupNum*5:1+groupNum*5+5]
                unitClass = groupInfo[1]
                avgHealth = groupInfo[2]
                numUnitsRemaining = groupInfo[4]
                # Create the units for the renderer
                unitGeom = None
                unitTranslate = rendering.Transform()
                if unitClass == 0:
                    unitGeom = rendering.make_circle(UNIT_SIZE)
                    unitGeom.add_attr(unitTranslate)
                elif unitClass == 1:
                    unitGeom = rendering.make_polygon([
                        (-UNIT_SIZE, -UNIT_SIZE), (-UNIT_SIZE, UNIT_SIZE),
                        (UNIT_SIZE, UNIT_SIZE), (UNIT_SIZE, -UNIT_SIZE)
                    ])
                    unitGeom.add_attr(unitTranslate)
                else:
                    unitGeom = rendering.make_polygon([
                        (0, UNIT_SIZE), (UNIT_SIZE, -UNIT_SIZE), (-UNIT_SIZE, -UNIT_SIZE)
                    ])
                    unitGeom.add_attr(unitTranslate)
                # Add the units to the renderer
                self.viewer.add_geom(unitGeom)
                if playerNum == 0:
                    self.p1UnitRenderInfo.append({
                        'geom': unitGeom,
                        'translate': unitTranslate,
                        'maxHealth': avgHealth,
                        'maxUnits': numUnitsRemaining,
                        'randomAngle': random.uniform(0.0, 360.0),
                        'randomTranslate': (random.uniform(-RANDOM_NOISE/2, RANDOM_NOISE/2), random.uniform(-RANDOM_NOISE/2, RANDOM_NOISE/2)),
                    })
                else:
                    self.p2UnitRenderInfo.append({
                        'geom': unitGeom,
                        'translate': unitTranslate,
                        'maxHealth': avgHealth,
                        'maxUnits': numUnitsRemaining,
                        'randomAngle': random.uniform(0.0, 360.0),
                        'randomTranslate': (random.uniform(-RANDOM_NOISE/2, RANDOM_NOISE/2), random.uniform(-RANDOM_NOISE/2, RANDOM_NOISE/2)),
                    })

    def getXYFromPolar(self, magnitude, angle):
        return (magnitude*math.cos(angle), magnitude*math.sin(angle))

    def updateUnitGroups(self):
        p2NodeCorrections = { 0:10, 1:7, 2:8, 3:9, 4:4, 5:5, 6:6, 7:1, 8:2, 9:3, 10:0 }

        for playerNum, unitRenderInfo in enumerate([self.p1UnitRenderInfo, self.p2UnitRenderInfo]):
            for groupNum in range(12):
                playerState = None
                if playerNum == 0:
                    playerState = self.player1State
                else:
                    playerState = self.player2State
                # Grab the unit group info
                groupInfo = playerState[1+groupNum*5:1+groupNum*5+5]
                nodeLoc = groupInfo[0] - 1
                if playerNum == 1:
                    nodeLoc = p2NodeCorrections[nodeLoc]
                unitClass = groupInfo[1]
                avgHealth = groupInfo[2]
                inTransit = groupInfo[3]
                numUnitsRemaining = groupInfo[4]
                # Update the unit group's render info
                nodeTranslation = self.p1NodeRenderInfo[nodeLoc]['translate'].translation

                if avgHealth == 0:
                    nodeTranslation = (-9000, 9000)

                randTransProp = unitRenderInfo[groupNum]['randomTranslate']
                if not inTransit:
                    unitRenderInfo[groupNum]['translate'].translation = (
                        nodeTranslation[0] + randTransProp[0],
                        nodeTranslation[1] + randTransProp[1]
                    )
                else:
                    cartXY = self.getXYFromPolar(CONTROL_POINT_SIZE + IN_TRANSIT_SPACE, unitRenderInfo[groupNum]['randomAngle'])
                    unitRenderInfo[groupNum]['translate'].translation = (
                        nodeTranslation[0] + cartXY[0],
                        nodeTranslation[1] + cartXY[1]
                    )
                if playerNum == 0:
                    if not inTransit:
                        unitRenderInfo[groupNum]['geom'].set_color(
                            0,
                            avgHealth / unitRenderInfo[groupNum]['maxHealth'] * 0.75,
                            avgHealth / unitRenderInfo[groupNum]['maxHealth'] * 0.75 + 0.25
                        )
                    else:
                        unitRenderInfo[groupNum]['geom'].set_color(0, 1.0, 1.0)
                else:
                    if not inTransit:
                        unitRenderInfo[groupNum]['geom'].set_color(
                            avgHealth / unitRenderInfo[groupNum]['maxHealth'] * 0.75 + 0.25,
                            avgHealth / unitRenderInfo[groupNum]['maxHealth'] * 0.75,
                            0
                        )
                    else:
                        unitRenderInfo[groupNum]['geom'].set_color(1.0, 1.0, 0)


    def setNodePositions(self):
        # Nodes 1 and 11
        x, y = 100, self.screen_height/2
        self.p1NodeRenderInfo[0]['translate'].set_translation(
            x, y
        )
        self.p1NodeRenderInfo[10]['translate'].set_translation(
            self.screen_width-x, y
        )
        # Rest of the nodes
        dx = (self.screen_width - 2 * x) / 4
        for i in range(1, 4):
            for j in range(1, 4):
                newX = x + i*dx
                newY = y + (j-2)*CONTROL_POINT_Y_SPACING
                self.p1NodeRenderInfo[j+3*(i-1)]['translate'].set_translation(
                    newX, newY
                )

    def updateNodes(self):
        p1NodeData = [self.getNodeData(i, 0) for i in range(11)]
        p2NodeData = [self.getNodeData(i, 1) for i in range(11)]
        
        for nodeNum in range(11):
            p1Node = p1NodeData[nodeNum]
            p2Node = p2NodeData[10-nodeNum]

            p1NodeControl = p1Node[3]
            absScaledP1NodeControl = min(
                abs(p1NodeControl),
                100
            ) / 100.0
            if p1NodeControl >= 0:
                self.p1NodeRenderInfo[nodeNum]['node'].set_color(
                    0, 0, absScaledP1NodeControl
                )
            else:
                self.p1NodeRenderInfo[nodeNum]['node'].set_color(
                    absScaledP1NodeControl, 0, 0
                )

    def render(self, mode):
        if self.viewer is None:
            self.viewer = rendering.Viewer(
                self.screen_width,
                self.screen_height
            )

            self.getLatestData()
            self.createNodesArray()
            self.setNodePositions()
            self.createNodeConnections()
            self.createUnitGroups()
            
        self.getLatestData()
        self.updateNodes()
        self.updateUnitGroups()
        time.sleep(TIME_BETWEEN_FRAMES)
        self.viewer.render(return_rgb_array=mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
