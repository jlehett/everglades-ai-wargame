import os
import numpy as np
import time
import json

# server stuff
import socket
import sys

class human_actions:

    def launch_server(self):
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        server_address = ('localhost', 10000)
        print("starting up on %s port %s" % server_address, file=sys.stderr)
        sock.bind(server_address)
        return sock

    def wait_for_player_input_on_tcp(self, sock):
        self.action = np.zeros(self.shape)

        # Listen for incoming connections
        sock.listen(1)

        while True:

            # empty input string
            clientInput = ""

            # Wait for a connection
            connection, client_address = sock.accept()

            try:
                print('waiting for client data', client_address, file=sys.stderr)
                fullDataString = ""
                # Receive the data in small chunks and retransmit it
                while True:
                    data = connection.recv(16)
                    print("recieved " % data, sys.stderr)
                    clientInput += str(data.decode('utf-8'))
                    #fullDataString = fullDataString + repr(data)

                    if data:
                        print('sending data back to the client', file=sys.stderr)
                        connection.sendall(data)
                        #print('nothing')
                    else:
                        print('no more data from', client_address)
                        #if clientInput[:9] == "actions: ":
                        print("parsing actions", file=sys.stderr)

                        #clientInput = clientInput[8:]                   # remove the "actions:" part of the string
                        #firstarray, secondarray = clientInput.split('|')

                        #firstarray = firstarray.split(',')            # split the input into (group,node,group,node,etc)
                        #secondarray = secondarray.split(',')

                        self.actions = np.zeros(self.shape)
                        #self.actions[:, 0] = firstarray
                        #self.actions[:, 1] = secondarray

                        self.actions = self.convertInputToActions(clientInput)
                        for i in range(0, self.num_actions):
                            print("Group " + str(self.actions[i][0]) + " is moving to Node " + str(self.actions[i][1]))

                        connection.sendall("turn success".encode('utf-8'))
                        return

            finally:
                # Clean up the connection
                connection.close()

    def __init__(self, action_space, player_num, map_name):
        self.action_space = action_space
        self.num_groups = 12

        self.actions = "" # output

        with open('./config/' + map_name) as fid:
            self.map_dat = json.load(fid)

        self.nodes_array = []
        for i, in_node in enumerate(self.map_dat['nodes']):
            self.nodes_array.append(in_node['ID'])

        self.num_nodes = len(self.map_dat['nodes'])
        self.num_actions = action_space

        self.shape = (self.num_actions, 2)

        self.unit_config = {
            0: [('controller',1), ('striker', 5)],# 6
            1: [('controller',3), ('striker', 3), ('tank', 3)],# 15
            2: [('tank',5)],# 20
            3: [('controller', 2), ('tank', 4)],# 26
            4: [('striker', 10)],# 36
            5: [('controller', 4), ('striker', 2)],# 42
            6: [('striker', 4)],# 46
            7: [('controller', 1), ('striker', 2), ('tank', 3)],# 52
            8: [('controller', 3)],# 55
            9: [('controller', 2), ('striker', 4)],# 61
            10: [('striker', 9)],# 70
            11: [('controller', 20), ('striker', 8), ('tank', 2)]# 100
        }

    def get_action(self, obs, sock):
        # Wait for a player connection and a message
        # HACK: this also launches an entire new server which is really shitty, figure the python way to split this out
        print("Waiting for player")

        self.wait_for_player_input_on_tcp(sock)

        self.actions = np.asarray(self.actions)
        #print('!!!!!!! Observation !!!!!!!!')
        #print(obs)
        #print(obs[0])
        #for i in range(45,101,5):
        #    print(obs[i:i+5])
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # action = np.zeros(self.shape)
        # action[:, 0] = np.random.choice(self.num_groups, self.num_actions, replace=False)
        # action[:, 1] = np.random.choice(self.nodes_array, self.num_actions, replace=False)
        # print('!!!actions!!!')
        # print(action)
        return self.actions

    def convertInputToActions(self, stringToParse):
        tokens = list()
        tokens = self.parseIntoTokens(stringToParse)
        return self.translateTokens(tokens)


    def parseIntoTokens(self, stringToParse):
        #need to parse out the 'b'
        tokens = list()
        tempToken = ""

        for i in range(0, len(stringToParse)):
            if((stringToParse[i] != " ") & (stringToParse[i] != ",") &(i != len(stringToParse) - 1)):
                if(stringToParse[i] == "'"):
                    if(i <= len(stringToParse)):
                        if(stringToParse[i+1] == "b"):
                            i = i + 2
                elif(stringToParse[i] == "b"):
                    if(i <= len(stringToParse)):
                        if(stringToParse[i+1] == "'"):
                            i = i + 1
                        else:
                            tempToken = tempToken + stringToParse[i]
                    else:
                        tempToken = tempToken + stringToParse[i]
                else:
                    if(stringToParse[i] != "'"):
                        tempToken = tempToken + stringToParse[i]
            elif(i == len(stringToParse)-1):
                if(stringToParse[i] != "'"):
                    tempToken = tempToken + stringToParse[i]
                tokens.append(tempToken)
            else:
                tokens.append(tempToken)
                tempToken = ""

        #for i in range(0, len(tokens)):
            #for j in range(0, len(tokens[i])):

        #for i in range(0, len(tokens)):
            #print("token "+str(i)+": "+tokens[i])


        return tokens

    def translateTokens(self, tokens):
        keywordGroup = "GROUP"
        keywordNode = "NODE"

        rows = self.num_actions
        cols = 2

        arrayOfActions = [[0 for i in range(cols)] for j in range(rows)]

        groupIndex = 0
        nodeIndex = 0
        searchingForNumber = False
        searchingForGroupNumber = False
        searchingForNodeNumber = False

        

        for i in range(0, len(tokens)):
            if((searchingForGroupNumber == False) & (searchingForNodeNumber == False)):
                if(groupIndex == nodeIndex):
                    if (str.upper(tokens[i]) == keywordGroup):
                        searchingForGroupNumber = True
                        #print("GROUP FOUND. searching for a Group Number Now")
                    #else:
                        #print("(1) " + tokens[i] + " is a filler word")
                elif(groupIndex > nodeIndex):
                    if(str.upper(tokens[i]) == keywordNode):
                        searchingForNodeNumber = True
                        #print("NODE FOUND. searching for a Node Number Now")
                    #else:
                        #print("tokens[i] is "+str.upper(tokens[i])+" and keywordNote is "+keywordNode)
                        #print("(2) " + tokens[i] + " is a filler word")
                #else:
                    #print("groupIndex < nodeIndex")
            elif(tokens[i].isdigit() == True):
                if(searchingForGroupNumber == True):
                    arrayOfActions[groupIndex][0] = tokens[i]
                    #print("Group "+str(groupIndex + 1)+" the token is "+ tokens[i])
                    searchingForGroupNumber = False
                    groupIndex = groupIndex + 1
                elif(searchingForNodeNumber == True):
                    #print("Node "+str(nodeIndex + 1)+" the token is "+ tokens[i])
                    arrayOfActions[nodeIndex][1] = tokens[i]
                    searchingForNodeNumber = False
                    nodeIndex = nodeIndex + 1
            #else:
                #print("(3) " + tokens[i] + " is a filler word")

        
        return arrayOfActions