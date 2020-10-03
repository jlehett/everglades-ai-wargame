from gym.envs.classic_control import rendering
import time

# Some adjustable constants for rendering
CONTROL_POINT_SIZE = 30
CONTROL_POINT_Y_SPACING = 150

class EvergladesRenderer:
    def __init__(self, game):
        self.game = game
        self.screen_width, self.screen_height = (900, 900)
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
        # Create player 2's node representations
        self.p2NodeRenderInfo = []
        for nodeNum in range(1, 12):
            node = rendering.make_circle(CONTROL_POINT_SIZE)
            translate = rendering.Transform()
            node.add_attr(translate)
            self.viewer.add_geom(node)
            self.p2NodeRenderInfo.append({
                'node': node,
                'translate': translate
            })

    def setNodePositions(self):
        # Nodes 1 and 11
        x, y = 100, 200
        self.p1NodeRenderInfo[0]['translate'].set_translation(
            x, y
        )
        self.p2NodeRenderInfo[0]['translate'].set_translation(
            x, self.screen_height-y
        )
        self.p1NodeRenderInfo[10]['translate'].set_translation(
            self.screen_width-x, y
        )
        self.p2NodeRenderInfo[10]['translate'].set_translation(
            self.screen_width-x, self.screen_height-y
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
                self.p2NodeRenderInfo[j+3*(i-1)]['translate'].set_translation(
                    newX, self.screen_height-newY
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
            p2NodeControl = p2Node[3]
            absScaledP2NodeControl = min(
                abs(p2NodeControl),
                100
            ) / 100.0
            if p2NodeControl >= 0:
                self.p2NodeRenderInfo[nodeNum]['node'].set_color(
                    absScaledP2NodeControl, 0, 0
                )
            else:
                self.p2NodeRenderInfo[nodeNum]['node'].set_color(
                    0, 0, absScaledP2NodeControl
                )

    def render(self, mode):
        if self.viewer is None:
            self.viewer = rendering.Viewer(
                self.screen_width,
                self.screen_height
            )

            self.createNodesArray()
            self.setNodePositions()
            
        self.getLatestData()
        self.updateNodes()
        time.sleep(0.0)
        self.viewer.render(return_rgb_array=mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None