import numpy as np
import matplotlib.pyplot as plt
from utils_ex10.odor_plotting import *
import time
import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
import pandas as pd
import seaborn as sns
from netgraph import Graph
from matplotlib.colors import LinearSegmentedColormap
import matplotlib


class APC_circuit:
    def __init__(
        self,
        time_ms=40000,
        learning_rate=0.000001,
        seed=1,
        n_PN = 50,
        n_PV = 4,
        n_SST = 8,
        n_VIP = 1,
    ):
        ### Store parameters passed by the function ###

        # Define the number of PN, SST & VIP neurons in the network
        self.n_PN = n_PN
        self.n_PV = n_PV
        self.n_SST = n_SST
        self.n_VIP = n_VIP

        # Size of input vector (number of distinct olfactory receptors)
        self.n_inputs = 10

        # Number of iterations (dt = 1 ms)
        self.time_ms = time_ms

        # Learning rate during weight update in Oja's rule
        self.learning_rate = learning_rate

        # Seed used for calls to numpy.random
        self.seed = seed
        np.random.seed(self.seed)

        # Set style for all ploting functions.
        plt.style.use('seaborn-v0_8-paper')
        self.color = ['#1f77b4', '#525252']

        ###############################################
        ### Set parameters for neurons and the network ###

        self.n_randInputs = 1
        self.con_n_PV_to_PN = 1  # Number of PV neurons that are connected to each PN at random. 23%

        # Conectivity parameters
        self.numberOfActiveInputs = 2  # Number of active inputs in the inputarray (Active Receptors / Glomeruli for a given odour so to speak).
        self.numberOfGlomeruliToPN = 2  # Number of Glomeruli each PN is connected to (assignemnt is random).

        self.con_n_SST_to_PN = 1  # Number of SST neurons that are connected to each PN at random.
        self.con_n_VIP_to_SST = 1  # Number of VIP neurons that are connected to each PN at random.

        self.p_PN = 0.4
        self.p_PV = 0.42  # Probability of a connection from a PN to an PV neuron (meaning that one PN is connected to ca. 30 % of all PVs)
        self.p_SST = 0.19  # Probability of a connection from a PN to an SST neuron (meaning that one PN is connected to ca. 19 % of all SSTs)
        # Population weights
        self.pop_weight_Input_PN = 2.7
        self.pop_weight_PN_PN = 0.07  # These weights are trainable
        self.pop_weight_PN_PV = 0.05
        self.pop_weight_PV_PN = 0.015
        self.pop_weight_PN_SST = 0.1
        self.pop_weight_SST_PN = 0.15
        self.pop_weight_VIP_SST = 4

        # Decay constant for EPSP
        self.k_PN = 0.0526  # Decay constant of PN membrane potential V_PN.
        self.k_SST = 0.1608  # Decay constant of the SST neuron membrane potential.
        self.k_VIP = 0.184  # Decay constant of the VIP neuron membrane potential.
        self.k_PV = 0.25  # Decay constant of the VIP neuron membrane potential.

        # Parameters for activation function
        # Maximal firing rates
        self.Rmax_PN = 150
        self.Rmax_PV = 193
        self.Rmax_SST = 199
        self.Rmax_VIP = 196

        # Resting Membrane potential for all neurons
        self.Vrest_PN = 0
        self.Vrest_PV = 0
        self.Vrest_SST = 0
        self.Vrest_VIP = 0

        # Thresholds for depolarization and firing of the neurons in units of membrane potential.
        self.Thresh_PN = 5
        self.Thresh_PV = 30
        self.Thresh_SST = 15
        self.Thresh_VIP = 5  # Set arbitrarly

        # Gains for ReLU function
        self.Gain_PN = 1
        self.Gain_PV = 2.7
        self.Gain_SST = 1.6
        self.Gain_VIP = 2.2  # Mean of SST and PV (no literature)

        ### Parameters for learning ###

        # Set parameters for OJA learning
        self.alpha = 0.8  # Rate of regularization in OJA.
        self.epsilon = 0  # Threshold for weight update.
        self.timeAvrg = 100  # Time over which is integrated in a learning step (Ojas rule)

        # Track outer products
        self.outr_act = np.zeros((self.timeAvrg, self.n_PN,
                                  self.n_PN))  # Stores the outer products of the activites. Used for integration in Ojas rule.
        ###############################################################

        ### Set time of odor presentation in milliseconds & gap between odor presentation
        self.n_burst = 100  # Number of bursts once the input fires // time the input is ON in ms.
        self.empty_burts = 2  # Gap in "n_bursts" between two odors (here 200 ms).
        self.scale = 0.004
        ###############################################################

        ### Initialize Membrane potentials and activities of all the neurons over time ###

        # Track the membrane potentials of neurons
        self.V_PN = np.zeros([self.n_PN, self.time_ms])
        self.V_PV = np.zeros([self.n_PV, self.time_ms])
        self.V_SST = np.zeros([self.n_SST, self.time_ms])
        self.V_VIP = np.zeros([self.n_VIP, self.time_ms])

        # Initialize with resting membrane potential
        self.V_PN[:, 0] = self.Vrest_PN
        self.V_PV[:, 0] = self.Vrest_PV
        self.V_SST[:, 0] = self.Vrest_SST
        self.V_VIP[:, 0] = self.Vrest_VIP

        # Track the activities of the neurons.
        self.Activity_PN = np.zeros([self.n_PN, self.time_ms])
        self.Activity_PV = np.zeros([self.n_PV, self.time_ms])
        self.Activity_SST = np.zeros([self.n_SST, self.time_ms])
        self.Activity_VIP = np.zeros([self.n_VIP, self.time_ms])

        ###############################################################

        ### Initialize vectors to store input and teacher signal

        self.teacher = np.zeros((self.n_VIP, self.time_ms))  # Stores the teacher signal over whole training time
        self.input = np.zeros((self.n_inputs, self.time_ms))  # Input array for each timestep
        self.RepInptHist = np.zeros(
            (self.n_randInputs, self.time_ms))  # Stores the position of the input that is presented repeatedly

        ###############################################################
        self._InputandTeacher()
        self._setConnectionsAndWeights()
        ### Used in plotting functions
        self.time_s = int(time_ms / 1000)

    def _setConnectionsAndWeights(self):
        '''
        Function initializes connection matrices for all the neurons using the probabilities defined in __init__.
        '''

        ### CONNECTIONS FOR UNTRAINABLE WEIGHTS
        # The connectivity matrix directly includes the constant weights.

        # SEED
        rng = np.random.default_rng(self.seed)

        # INPUT
        # Set connections and strength from input stream to PN neurons. Here the weights are included into the connection matrix.
        A = np.zeros((self.n_PN, self.n_inputs))
        A[:, :self.numberOfGlomeruliToPN] = 1
        self.weightInput_PN = rng.permuted(A, axis=1) * np.random.uniform(0, 1, size=(self.n_PN, self.n_inputs))
        n_weights = self.n_PN * self.numberOfGlomeruliToPN
        Inpt_PN_factor = np.sum(self.weightInput_PN) / n_weights
        self.weightInput_PN = (self.pop_weight_Input_PN / Inpt_PN_factor) * self.weightInput_PN

        # PN TO PV
        # Set connecntions from PN to PV
        self.weightPN_PV = np.random.choice([0, 1], size=(self.n_PV, self.n_PN), p=[1 - self.p_PV, self.p_PV])
        self.weightPN_PV = self.weightPN_PV * self.pop_weight_PN_PV  # Since the average weight is equal to 1, does only work as long as all weights the same!

        # PN TO SST
        # Set connections from PN to SST
        self.weightPN_SST = np.random.choice([0, 1], size=(self.n_SST, self.n_PN), p=[1 - self.p_SST, self.p_SST])
        self.weightPN_SST = self.weightPN_SST * self.pop_weight_PN_SST  # Since the average weight is equal to 1, does only work as long as all weights the same!

        # VIP TO SST
        # Set connections and strength from VIP to SST neurons. Here the weights are included into the connection matrix.
        A = np.zeros((self.n_SST, self.n_VIP))
        A[:, :self.con_n_VIP_to_SST] = 1
        self.weightVIP_SST = rng.permuted(A, axis=1)
        VIP_SST_factor = np.sum(self.weightVIP_SST)
        self.weightVIP_SST = self.weightVIP_SST * (self.pop_weight_VIP_SST / VIP_SST_factor)

        ### CONNECTIONS FOR TRAINABLE WEIGHTS
        # The connectivity matrix does not include weights.

        # RECURRENT PN TO PN CONNECTIONS

        # PN TO PN
        # Set recurrent connections from PN to PN
        self.conPN_PN = np.random.choice([0, 1], size=(self.n_PN, self.n_PN), p=[1 - self.p_PN, self.p_PN])
        np.fill_diagonal(self.conPN_PN,
                         0)  # Set weights for recurrent connections back to the same neuron to zero (Not possible to train using Ojas rule)

        # Allocate memory for weight matrix and randomly initialize weights
        self.weightPN_PN = np.zeros((self.time_ms, self.n_PN, self.n_PN))
        self.weightPN_PN[0, :, :] = np.random.uniform(0, 1, size=(self.n_PN, self.n_PN)) * self.conPN_PN
        n_weights = np.sum(self.conPN_PN)
        PN_PN_factor = self.pop_weight_PN_PN / (np.sum(self.weightPN_PN[0, :, :]) / n_weights)
        self.weightPN_PN[0, :, :] = self.weightPN_PN[0, :,
                                    :] * PN_PN_factor  # Weights are normalized such that the population average strength, independent of the number of connecitons, is 7 (Paper: Differential Excitability of PV and SST Neurons Results in...

        # SST TO PN
        # Set connections from SST to PN
        A = np.zeros((self.n_PN, self.n_SST))
        A[:, :self.con_n_SST_to_PN] = 1
        self.weightSST_PN = rng.permuted(A, axis=1)

        # PV TO PN
        # Set connections from PV to PN
        A = np.zeros((self.n_PN, self.n_PV))
        A[:, :self.con_n_PV_to_PN] = 1
        self.weightPV_PN = rng.permuted(A, axis=1)

        # SST to PN
        self.weightSST_PN = self.weightSST_PN * self.pop_weight_SST_PN
        # PV to PN
        self.weightPV_PN = self.weightPV_PN * self.pop_weight_PV_PN

    def _InputandTeacher(self):
        '''
        Creates the input and teaching signal.

        '''

        if (int(self.time_ms) % int(self.n_burst)) != 0:
            print("Simulation time needs to be divisible by burst size")
            exit()

        if (self.n_inputs < self.numberOfActiveInputs):
            print("The size of the input vector needs to be larger than the number of active inputs.")
            exit()

        # Create two distinct input vectors.
        x1, x2 = np.zeros(self.n_inputs), np.zeros(self.n_inputs)
        x1[0:self.numberOfActiveInputs] = 1
        x2[self.numberOfActiveInputs:2 * self.numberOfActiveInputs] = 1

        # Variable used to alternate between the three different inputs (x1, x2 and rand(x2))
        alternate = 1
        for i in range(self.time_ms):

            # Generate the input with teaching signal
            if i % ((self.empty_burts + 1) * self.n_burst) == 0 and alternate == 1:
                for j in range(self.n_burst):
                    self.input[:, i + j] = x2
                    self.teacher[0, i + j] = 10
                alternate = 2
                continue

            # Generate a random input to present between the two recurring once.
            if i % ((self.empty_burts + 1) * self.n_burst) == 0 and alternate == 2:
                rand = np.random.permutation(x2)
                for j in range(self.n_burst):
                    self.input[:, i + j] = rand
                alternate = 3
                continue

            # Generate the periodic random input
            if i % ((self.empty_burts + 1) * self.n_burst) == 0 and alternate == 3:
                for j in range(self.n_burst):
                    self.input[:, i + j] = x1
                    self.RepInptHist[0, i + j] = 1
                alternate = 1
                continue

    def getEnsembles(self):
        '''
        The function extracts neuronal ensembles for each odor presented repeatedly.
        '''
        # Neurons that show activity at the very last input presentation and are not decreasing in activity are counted as ensembles.

        min_activity = 10
        # max_negativeSlope = -1
        # timewindow_Slope = 15

        self.ensembleNeurons = []
        self.ensembleActivites = []
        self.ensembleActivitesUntrained = []
        self.ensembleNeuronsUntrained = []

        # Get ensemble neurons and activities where teacher signal is present.
        for i in range(self.n_VIP):
            # Get indices (of timepoint) of untraind (idx0) and trained (idx).
            idx0 = np.where(self.teacher[i, :].astype(bool) == True)[0][
                self.n_burst - 1]  # Get index of first presentation of corresponding VIP input (index of last input presentation in first burst)
            idx = np.where(self.teacher[i, :].astype(bool) == True)[0][
                -1]  # Get index of last presentation of corresponding VIP input

            # Get Neuron numbers and activities after training
            self.ensembleNeurons.append(np.where(self.Activity_PN[:, idx] > min_activity)[
                                            0].tolist())  # Append indices (neuron number) for each ensemble. & ((self.Activity_PN[:,idx] - self.Activity_PN[:,idx-1]) > 0) #& (np.average(np.diff(self.Activity_PN[:,idx-timewindow_Slope:idx]),axis = 1) > max_negativeSlope)
            self.ensembleActivites.append(self.Activity_PN[self.ensembleNeurons[
                -1], idx].tolist())  # Append activities of the neurons in ensemble.

            # Get Neuron numbers and activities before training
            self.ensembleActivitesUntrained.append(self.Activity_PN[self.ensembleNeurons[
                -1], idx0].tolist())  # Append activities of the untrained neurons in ensemble.

        for j in range(self.n_randInputs):
            # Get ensemble neurons and activities for repeatly presented input w/o teacher signal. It is stored as the last list.
            idx0 = np.where(self.RepInptHist[j, :].astype(bool) == True)[0][self.n_burst - 1]
            idx = np.where(self.RepInptHist[j, :].astype(bool) == True)[0][
                -1]  # Get index of last presentation of corresponding VIP input
            self.ensembleNeurons.append(np.where(self.Activity_PN[:, idx] > min_activity)[
                                            0].tolist())  # Append indices (neuron number) for each ensemble. #& ((self.Activity_PN[:,idx] - self.Activity_PN[:,idx-1]) > 0)
            self.ensembleActivites.append(self.Activity_PN[self.ensembleNeurons[
                -1], idx].tolist())  # Append activities of the neurons in ensemble.

            self.ensembleActivitesUntrained.append(self.Activity_PN[self.ensembleNeurons[
                -1], idx0].tolist())  # Append activities of the untrained neurons in ensemble.


    def feedForward(self, i):
        ### Dimensions of variables:
        # self.weightInput_PN: (n_PN x n_inputs)
        # self.weightPN_PN: (time,n_PN,n_PN)
        # self.weightPV_PN: (n_PN,n_PV)
        # self.weightPN_PV: (n_PV,n_PN)
        # self.weightPN_SST: (n_SST,n_PN)
        # self.weightVIP_SST: (n_SST,n_VIP)
        # self.input: (n_inputs, time_ms)
        # self.V_X: (n_X, time_ms), X = {PN, PV, SST, VIP}
        # self.Activity_X: (n_X, time_ms), X = {PN, PV, SST, VIP}

        ### Other variables
        # i indexes the time.
        # self.teacher: (n_VIP, time_ms)
        # self.k_X: (single value)

        ### Simulate membrane potential of PN, SST, PV & VIP neurons. ###

        # Calculate the activation by the input as well as the recurrent activity (rec_act) to use below.
        Input = (self.weightInput_PN @ self.input[:, i])
        rec_act = (self.scale * self.weightPN_PN[i - 1, :, :] @ self.Activity_PN[:, i - 1])

        # PN: V_PN(t-1) + Input + recurrent connection - self decay - SST inhibition (max equal to odor input (distal)) - PV inhibition
        self.V_PN[:, i] = self.V_PN[:, i - 1] + Input + rec_act - (self.k_PN * self.V_PN[:, i - 1]) - np.minimum(Input,
                                                                                                                 self.weightSST_PN @ self.Activity_SST[
                                                                                                                                     :,
                                                                                                                                     i - 1]) - (
                                      self.weightPV_PN @ self.Activity_PV[:, i - 1])

        # PV: V_PV(t-1) + activation by PN - selfdecay #- rec_inhibition by PV
        self.V_PV[:, i] = self.V_PV[:, i - 1] + (self.weightPN_PV @ self.Activity_PN[:, i - 1]) - self.k_PV * self.V_PV[:,
                                                                                                              i - 1]

        # SST: V_SST(t-1)  + activation by PN - self decay - VIP inhibition
        self.V_SST[:, i] = self.V_SST[:, i - 1] + (self.weightPN_SST @ self.Activity_PN[:, i - 1]) - (
                    self.k_SST * self.V_SST[:, i - 1]) - (self.weightVIP_SST @ self.Activity_VIP[:, i - 1])

        # VIP: V_VIP(t-1) + teacher - self decay
        self.V_VIP[:, i] = self.V_VIP[:, i - 1] - (self.V_VIP[:, i - 1] * self.k_VIP) + self.teacher[:, i]

        # Convert to activity using the capped ReLU function
        self.Activity_PN[:, i] = np.maximum(0, np.minimum(self.Gain_PN * (self.V_PN[:, i] - self.Thresh_PN), self.Rmax_PN))
        self.Activity_PV[:, i] = np.maximum(0, np.minimum(self.Gain_PV * (self.V_PV[:, i] - self.Thresh_PV), self.Rmax_PV))
        self.Activity_SST[:, i] = np.maximum(0, np.minimum(self.Gain_SST * (self.V_SST[:, i] - self.Thresh_SST),
                                                           self.Rmax_SST))
        self.Activity_VIP[:, i] = np.maximum(0, np.minimum(self.Gain_VIP * (self.V_VIP[:, i] - self.Thresh_VIP),
                                                           self.Rmax_VIP))

    def simulate(self):
        # dt = 1ms
        # number of ms to simulate is stored in variable: self.time_ms

        st = time.time()  # Timestamp at start of simulation

        # Actual simulation
        for i in range(1, self.time_ms):
            self.feedForward(i)

        et = time.time()  # Timestamp at end of simulation
        # print('Simulation complete after', et - st, 'seconds')  # Print time duration of simulation

    def hebbian(self, i):
        d_weight = self.learning_rate * np.outer(self.Activity_PN[:, i], self.Activity_PN[:, i - 1])
        self.weightPN_PN[i, :, :] = np.maximum((d_weight * self.conPN_PN) + self.weightPN_PN[i - 1, :, :],
                                               0)  # Only update weights that really exist and only if they stay positive.

    def simulate_hebbian(self):
        # dt = 1ms
        # number of ms to simulate is stored in variable: self.time_ms

        st = time.time()  # Timestamp at start of simulation

        # Actual simulation
        for i in range(1, self.time_ms):
            self.feedForward(i)
            self.hebbian(i)

        et = time.time()  # Timestamp at end of simulation
        # print('Simulation complete after', et - st, 'seconds')  # Print time duration of simulation

    def _Oja(self, i):
        j = min(i, self.timeAvrg)  # Stores the indicies for the Integration

        self.outr_act[i % self.timeAvrg, :, :] = np.outer(self.Activity_PN[:, i], self.Activity_PN[:,
                                                                                  i - 1])  # Stores outer products in a circular buffer fashion, over the self.timeAvrg time points (Hebbian part of Oja rule)

        # Calculate integral
        int_1 = (1 / j) * np.sum(self.outr_act,
                                 axis=0)  # Calculates the sum over "timeAvrg" for the input*output matrix. Axis = 0 makes sure to sum over the time dimension.

        avr_output_sq = np.average(self.Activity_PN[:, i - j:i],
                                   axis=1) ** 2  # Calculate the normalizing part of Ojas rule (squared average over time (TIME OJA))

        int_2 = avr_output_sq[:, np.newaxis] * self.weightPN_PN[i - 1, :, :]

        d_weight = self.learning_rate * (int_1 - self.alpha * int_2)  # Finally, calculate the weight update.
        self.weightPN_PN[i, :, :] = np.maximum((d_weight * self.conPN_PN) + self.weightPN_PN[i - 1, :, :],
                                               0)  # Only update weights that really exist and only if they stay be positive.

    def simulate_oja(self):
        # dt = 1ms
        # number of ms to simulate is stored in variable: self.time_ms
        st = time.time()  # Timestamp at start of simulation

        # Actual simulation
        for i in range(1, self.time_ms):
            self.feedForward(i)
            self._Oja(i)

        et = time.time()  # Timestamp at end of simulation
        # print('Simulation complete after', et - st, 'seconds')  # Print time duration of simulation

    def plotNetwork(self, log_weights=False):
        ### General stuff

        # Generate weight matrix (weight between neurons are summed, eg. weight between PN 1 and 4 == 4 and 1)
        weight_matrix = np.zeros((self.n_PN, self.n_PN))
        for i in range(self.n_PN):
            for j in range(self.n_PN):
                weight_matrix[i, j] = self.weightPN_PN[-1, i, j] + self.weightPN_PN[-1, j, i]

        # normalize
        # weight_matrix = weight_matrix / np.max(weight_matrix)

        if log_weights:
            weight_matrix = np.log(weight_matrix + 1)

        ### Generate Graph
        G = nx.from_numpy_array(weight_matrix)

        # Set node shape
        node_shape = {i: 'o' for i in G.nodes()}

        node_edge_width = {i: 0.5 for i in G.nodes()}
        node_edge_color = {i: 'black' for i in G.nodes()}
        ### set colors
        # If a neuron belongs to different ensembles, it gets colored according to the ensemble it showes the strongest activity in.
        # Only checks pair wise and not if eg it is in all three lists.

        color = {}
        for i in G.nodes():
            color[i] = '#ede8e4'

        for i in self.ensembleNeurons[0]:
            color[i] = self.color[0]

        # Check if neurons belong to multiple ensembles.
        overlay = 0
        for i in self.ensembleNeurons[1]:
            if i in self.ensembleNeurons[0]:
                color[i] = '#6C7D8A'
                overlay = 1
            else:
                color[i] = self.color[1]

        # DONE COLOR
        self.colors = color

        weight = nx.get_edge_attributes(G, 'weight')

        edges = G.edges()
        max_width = max(weight.values())
        # max_width = 4
        width = {key: weight[key] / max_width for key in weight}  # Used to draw the thickness of the connections (needs to be scaled down a bit to not be too thick...)

        # Generate edge colors
        norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(weight_matrix))
        # Choose colormap
        # cmap = pl.cm.autumn_r
        cmap = matplotlib.cm.autumn_r
        edgeCol = {key: cmap(norm(weight[key])) for key in weight.keys()}

        # Code to add an increasing alpha value in addition to the color map
        '''
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:,-1] = np.linspace(0, 1, cmap.N) # Set alpha
        # Create new colormap
        my_cmap = ListedColormap(my_cmap)
        # Create color 'edgeCol'
        edgeCol = {key: my_cmap(norm(weight[key])) for key in weight.keys()}
        '''

        node_proxy_artists = []
        # Create proxy artists for legend handles.
        if overlay == 1:
            legend_color = ['#1f77b4', '#7f7f7f', '#6C7D8A']
            legend_label = ['teacher', 'control', 'overlay']
            for node in range(self.n_VIP + 2):
                proxy = plt.Line2D([], [], linestyle='None', color=legend_color[node], label=legend_label[node],
                                   marker=node_shape[node], markersize=8)
                node_proxy_artists.append(proxy)

        else:
            legend_color = self.color
            legend_label = ['teacher', 'control']
            for node in range(self.n_VIP + 1):
                proxy = plt.Line2D([], [], linestyle='None', color=legend_color[node], label=legend_label[node],
                                   marker=node_shape[node], markersize=8)
                node_proxy_artists.append(proxy)

        # Calculate the position of nodes
        pos = nx.spring_layout(G, scale=0.5)

        # Plot graph
        fig, ax = plt.subplots(figsize=(20, 10))
        g = Graph(list(edges),
                  node_layout=pos,
                  node_layout_kwargs=dict(edge_length=weight),
                  ax=ax,
                  edge_width=width,
                  edge_color=edgeCol,
                  node_size=2,
                  node_labels=True,
                  node_color=color,
                  node_edge_width=node_edge_width,
                  node_edge_color=node_edge_color
                  )

        ax.set_aspect('equal')

        # Add legend
        node_legend = ax.legend(handles=node_proxy_artists, fontsize=12, framealpha=0)
        ax.add_artist(node_legend)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        clb = plt.colorbar(sm, ax=ax, shrink=0.5)

        if log_weights:
            clb.ax.set_title('log Weights')
        else:
            clb.ax.set_title('Weights')

        plt.show()

    def plotActivityPub(self):
        '''
        Plots the input, evolution of weights and evolution of output/activity in one plot.
        For "save == True" a subfolder is created (see documentation in _makeDir()) and the generated plots are saved there. Further,
        a .txt file is saved containing all parameters used for initializaiton of the neuron as well as some additional information.
        '''
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        Llim = (-0.1, 0.9)
        Rlim = (self.time_s - 1, self.time_s)
        d = .015  # how big to make the diagonal lines in axes coordinates
        mypad = 0.4
        ####################################################

        time = np.linspace(0, self.time_s, num=self.time_ms)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, gridspec_kw={'height_ratios': [6, 2, 2, 2, 1]},
                                                      sharex=True,
                                                      figsize=(10, 6))

        # Divide plot into two parts (broken x-axis)
        divider = make_axes_locatable(ax1)
        ax1R = divider.new_horizontal(size="100%", pad=mypad)
        fig.add_axes(ax1R)

        # plot the same data on both axes
        # Plot PN
        for j in range(self.n_PN):
            ax1.plot(time, self.Activity_PN[j, :], linewidth=1)
            ax1R.plot(time, self.Activity_PN[j, :], linewidth=1)

        # Limit in seconds
        ax1.set_xlim(Llim)
        ax1R.set_xlim(Rlim)

        ax1R.set_xticklabels([])
        ax1R.set_yticklabels([])

        ax1.set_title("PN", y=1.0, pad=-9)
        ax1R.set_title("PN", y=1.0, pad=-9)

        # hide the spines between ax and ax2
        # ax1.spines['right'].set_visible(False)
        # ax1R.spines['left'].set_visible(False)
        # ax1.tick_params(labelright='off')

        '''
        # Plot PN
        for j in range(self.n_PN):
            ax1.plot(time,self.Activity_PN[j,:])
        ax1.set_title("PN",y=1.0, pad=-14)
        # Decrease width of blox to leave space for legend.
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        '''

        # Divide plot into two parts (broken x-axis)
        divider = make_axes_locatable(ax2)
        ax2R = divider.new_horizontal(size="100%", pad=mypad)
        fig.add_axes(ax2R)

        # Plot SST
        for j in range(self.n_SST):
            ax2.plot(time, self.Activity_SST[j, :], linewidth=1)
            ax2R.plot(time, self.Activity_SST[j, :], linewidth=1)

        ax2.set_title("SST", y=1.0, pad=-9)
        ax2R.set_title("SST", y=1.0, pad=-9)
        # Decrease width of blox to leave space for legend.
        # = ax2.get_position()
        # ax2.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Limit in seconds
        ax2.set_xlim(Llim)
        ax2R.set_xlim(Rlim)

        ax2R.set_xticklabels([])
        ax2R.set_yticklabels([])

        # Plot PV

        # Divide plot into two parts (broken x-axis)
        divider = make_axes_locatable(ax3)
        ax3R = divider.new_horizontal(size="100%", pad=mypad)
        fig.add_axes(ax3R)

        for j in range(self.n_PV):
            ax3.plot(time, self.Activity_PV[j, :], linewidth=1)
            ax3R.plot(time, self.Activity_PV[j, :], linewidth=1)

        ax3.set_title("PV", y=1.0, pad=-9)
        ax3R.set_title("PV", y=1.0, pad=-9)
        '''	
        ax3.set_title("PV",y=1.0, pad=-12)
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        '''

        # Limit in seconds
        ax3.set_xlim(Llim)
        ax3R.set_xlim(Rlim)

        ax3R.set_xticklabels([])
        ax3R.set_yticklabels([])

        # Plot VIP

        # Divide plot into two parts (broken x-axis)
        divider = make_axes_locatable(ax4)
        ax4R = divider.new_horizontal(size="100%", pad=mypad)
        fig.add_axes(ax4R)

        for j in range(self.n_VIP):
            ax4.plot(time, self.Activity_VIP[j, :], label="VIP " + str(j + 1), linewidth=1)
            ax4R.plot(time, self.Activity_VIP[j, :], label="VIP " + str(j + 1), linewidth=1)

        # ax4.set_title("VIP",y=1.0, pad=-12)

        # Decrease width of blox to leave space for legend.
        # box = ax4.get_position()
        # ax4.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # leg = ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Put a legend to the right of the current axis

        # change the line width for the legend
        # for line in leg.get_lines():
        #	line.set_linewidth(1.5)

        # Limit in seconds
        ax4.set_xlim(Llim)
        ax4R.set_xlim(Rlim)
        ax4R.set_xticklabels([])
        ax4R.set_yticklabels([])

        ax4.set_title("VIP", y=1.0, pad=-9)
        ax4R.set_title("VIP", y=1.0, pad=-9)

        myinput = np.sum(self.input, axis=0) / 2

        '''
        for i in range(self.n_inputs):
            ax5.vlines(time, np.zeros(self.time_ms), myinput,linewidth=0.4, color='gray', label = "input")
        '''

        # rainbow = ax5._get_lines.prop_cycler

        teacher1 = np.zeros(self.time_ms)
        randinpt = np.zeros(self.time_ms)
        for i in range(self.time_ms):
            if myinput[i] == 1 and self.teacher[0, i] == 1:
                teacher1[i] = 1
            elif myinput[i] == 1 and self.teacher[0, i] != 1:
                randinpt[i] = 1

        # Plot VIP

        # Divide plot into two parts (broken x-axis)
        divider = make_axes_locatable(ax5)
        ax5R = divider.new_horizontal(size="100%", pad=mypad)
        fig.add_axes(ax5R)

        ax5.vlines(time, np.zeros(self.time_ms), teacher1, linewidth=0.4, color='#4C72B0', label="input + teacher 1")
        ax5.vlines(time, np.zeros(self.time_ms), randinpt, linewidth=0.4, color='gray', label="input w/o teacher")

        ax5R.vlines(time, np.zeros(self.time_ms), teacher1, linewidth=0.4, color='#4C72B0', label="input + teacher 1")
        ax5R.vlines(time, np.zeros(self.time_ms), randinpt, linewidth=0.4, color='gray', label="input w/o teacher")

        # Limit in seconds
        ax5.set_xlim(Llim)
        ax5.set_ylim(0, 2)
        ax5R.set_xlim(Rlim)
        ax5R.set_ylim(0, 2)
        ax5R.set_yticklabels([])

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax5.plot(1, 0, transform=ax5.transAxes, **kwargs)
        ax5R.plot(0, 0, transform=ax5R.transAxes, **kwargs)

        ax5.set_title("input", y=1.0, pad=-9)
        ax5R.set_title("input", y=1.0, pad=-9)
        '''
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax5.transAxes, color='k', clip_on=False)
        ax5.plot((1-d, 1+d), (-d, +d), **kwargs)
        #ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

        kwargs.update(transform=ax5R.transAxes)  # switch to the bottom axes
        #ax1R.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax5R.plot((-d, +d), (-d, +d), **kwargs)

        '''
        '''
        # Shrink current axis by 20%
        box = ax5.get_position()
        ax5.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # Put a legend to the right of the current axis
        leg = ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))



        #leg = ax5.legend()
        ax5.set_title("Input",y=1, pad=-12)
        ax5.set_xlabel("time (s)",fontsize=14)
        ax5.set_ylim((0,2))



        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(1.5)

        '''
        fig.text(0.07, 0.5, 'activity (rate)', va='center', rotation='vertical', fontsize=14)
        fig.text(0.49, 0.05, 'time (s)', va='center', rotation='horizontal', fontsize=14)
        # ax1.text(0.5, 1,1, 'untrained')
        # fig.text(0.685, 0.9, 'trained', va='center', rotation='horizontal',fontsize=14,verticalalignment='top',horizontalalignment='center')

        plt.show()

    def plotConnectivity(self):
        colors = ["lightgray", "gray"]
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))

        # PN to PN connection
        fig, ax = plt.subplots()
        sns.heatmap(self.conPN_PN, ax=ax, cmap=cmap, square=True, linewidths=.5, cbar_kws={"shrink": .1, "aspect": 2})
        ax.set_title("PN to PN")
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.25, 0.75])
        colorbar.set_ticklabels(['not connected', 'connection'])

        plt.show()

        '''
        # PN to PV and PN to SST connection
        fig, (ax1, ax2) = plt.subplots(nrows = 2)
        sns.heatmap(self.conPN_PV, ax=ax1, cmap=cmap, square=True,  linewidths=.5, cbar = False)
        sns.heatmap(self.conPN_SST, ax=ax2, cmap=cmap, square=True,  linewidths=.5, cbar = False)
        ax1.set_title("PN to PV ")
        ax2.set_title("PN to SST")

        plt.show()


        # PV to PN and SST to PN
        fig, (ax1, ax2) = plt.subplots(ncols = 2)
        sns.heatmap(self.conPV_PN, ax=ax1, cmap=cmap, square=True,  linewidths=.5, cbar = False)
        sns.heatmap(self.conSST_PN, ax=ax2, cmap=cmap, square=True,  linewidths=.5, cbar = False)
        ax1.set_title("PV to PN ")
        ax2.set_title("SST to PN")

        plt.show()


        # VIP to SST and SST to PV and PV to PV
        fig, (ax1) = plt.subplots(ncols = 1)
        sns.heatmap(self.conVIP_SST, ax=ax1, cmap=cmap, square=True,  linewidths=.5, cbar = False)
        #sns.heatmap(self.conSST_PV, ax=ax2, cmap=cmap, square=True,  linewidths=.5, cbar = False)
        #sns.heatmap(self.conPV_PV, ax=ax3, cmap=cmap, square=True,  linewidths=.5, cbar = False)
        ax1.set_title("VIP to SST ")



        plt.show()
        '''

    def plotEnsembleWeights(self, log_weights=False):
        from scipy.stats import sem
        # Number of repeatedly presented input w/o teacher signal
        import matplotlib as mpl
        mpl.rcParams['path.simplify'] = True
        mpl.rcParams['path.simplify_threshold'] = 1.0

        # Store Means and standard deviation of within ensemble weight evolution
        self.withinWeights_Mean = np.zeros((self.n_VIP + self.n_randInputs, self.time_ms))
        self.withinWeights_SD = np.zeros((self.n_VIP + self.n_randInputs, self.time_ms))

        # Store Means and standard deviation of between ensemble weight evolution
        self.betweenWeights_Mean = np.zeros(self.time_ms)
        self.betweenWeights_SD = np.zeros(self.time_ms)

        # Store the indexes of neurons that are not in an ensemble.
        tempConnMatrix_Between = np.full((self.n_PN, self.n_PN), False)
        # Set all connections between the neurons in an ensemble to True, without allowing recurrent connections.
        for VIP in range(self.n_VIP + self.n_randInputs):

            # Temp matrix used to store index of ensemble connections (used to index weight matrix.)
            tempConnMatrix = np.full((self.n_PN, self.n_PN), False)

            # Loop trough all possible connections in the ensemble
            for i in self.ensembleNeurons[VIP]:
                for j in self.ensembleNeurons[VIP]:
                    # Check if the neurons are really connected:
                    if self.conPN_PN[i, j] == 1:
                        tempConnMatrix[i, j] = True

            # Calculate the mean and SD for each ensemble.
            self.withinWeights_Mean[VIP, :] = np.average(self.weightPN_PN[:, tempConnMatrix], axis=1)

            # self.withinWeights_SD[VIP,:] = np.std(self.weightPN_PN[:,tempConnMatrix],axis=1)
            self.withinWeights_SD[VIP, :] = sem(self.weightPN_PN[:, tempConnMatrix], axis=1)

        # Get weight evolution between two ensembles (only works for two VIPs at the time...)
        tempConnMatrix = np.full((self.n_PN, self.n_PN), False)
        for i in self.ensembleNeurons[0]:
            for j in self.ensembleNeurons[1]:
                # Check that neurons are connected and not both present in one of the ensembles
                if self.conPN_PN[i, j] == 1 and (j not in self.ensembleNeurons[0]) and (
                        i not in self.ensembleNeurons[1]):
                    tempConnMatrix[i, j] = True

        self.betweenWeights_Mean = np.average(self.weightPN_PN[:, tempConnMatrix], axis=1)

        # self.betweenWeights_SD = np.std(self.weightPN_PN[:,tempConnMatrix],axis=1)
        self.betweenWeights_SD = sem(self.weightPN_PN[:, tempConnMatrix], axis=1)
        if self.n_VIP == 2:
            name = [' A ', ' B ', ' A ', ' B ']
        else:
            name = [' ', ' ', ' ', ' ']
        # Plot the results...

        fig, ax = plt.subplots(figsize=(11, 8))
        for VIP in range(self.n_VIP + self.n_randInputs):

            if VIP >= self.n_VIP:
                ax.plot(np.linspace(0, self.time_s, num=self.time_ms), self.withinWeights_Mean[VIP, :],
                        label='within control' + name[VIP] + r'$\pm$ SEM', color=self.color[VIP])
            else:
                ax.plot(np.linspace(0, self.time_s, num=self.time_ms), self.withinWeights_Mean[VIP, :],
                        label='within teacher' + name[VIP] + r'$\pm$ SEM', color=self.color[VIP])
            ax.fill_between(np.linspace(0, self.time_s, num=self.time_ms),
                            self.withinWeights_Mean[VIP, :] - self.withinWeights_SD[VIP, :],
                            self.withinWeights_Mean[VIP, :] + self.withinWeights_SD[VIP, :], color=self.color[VIP],
                            alpha=0.3)

        # Plot the between weight evolution (not quite right)
        ax.plot(np.linspace(0, self.time_s, num=self.time_ms), self.betweenWeights_Mean, color='black',
                linestyle='dashed',
                label=r'between $\pm$ SEM')
        ax.fill_between(np.linspace(0, self.time_s, num=self.time_ms),
                        self.betweenWeights_Mean - self.betweenWeights_SD,
                        self.betweenWeights_Mean + self.betweenWeights_SD, color='black', alpha=0.3)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.8)
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel("time (s)", fontsize=14)
        ax.set_ylabel("average weight", fontsize=14)
        # ax.set_ylim((0, 2))
        if log_weights:
            ax.set_yscale('log')
        yticks = ax.yaxis.get_major_ticks()
        yticks[-1].set_visible(False)

        plt.show()

        mpl.rcParams['path.simplify_threshold'] = 0.1111

    def plotEnsembleStrength(self):
        # Generate lists to store the name of ensemble and if ensemlbes are trained or not.
        Ensembles = []
        Trained = []
        Activities = []
        Neuron = []

        col = ['#98df8a', '#2ca02c']
        # Create data frame for untrained examples.
        Activities = self.ensembleActivitesUntrained.copy() + self.ensembleActivites.copy()
        Neuron = self.ensembleNeurons.copy() + self.ensembleNeurons.copy()
        Trained = [['untrained' for i in range(len(sublist))] for sublist in self.ensembleActivitesUntrained.copy()] + [
            ['trained' for i in range(len(sublist))] for sublist in self.ensembleActivites.copy()]
        # This is a concat. of two very long (but identical) list comprehensions. Creates a list with the same dimensions as 'Activities' or 'Neuron' but stores if a teacher signal is present (VIP) or the random input is presented.

        Ensembles = [['teacher' for i in range(len(sublist[1]))] if (
                sublist[0] < len(self.ensembleActivitesUntrained.copy()) - 1) else ['control' for i in
                                                                                    range(len(sublist[1]))] for sublist
                     in enumerate(self.ensembleActivitesUntrained.copy())] \
                    + [['teacher' for i in range(len(sublist[1]))] if (
                sublist[0] < len(self.ensembleActivites.copy()) - 1) else ['control' for i in range(len(sublist[1]))]
                       for sublist in enumerate(self.ensembleActivites.copy())]

        # Uncomment if more than one teacher present!!!! And comment out two lines above!
        # Ensembles = [['teacher ' + str(sublist[0] + 1) for i in range(len(sublist[1]))] if (sublist[0] < len(self.ensembleActivitesUntrained.copy()) - 1) else ['no teacher' for i in range(len(sublist[1]))] for sublist in enumerate(self.ensembleActivitesUntrained.copy()) ] \
        # + [['teacher ' + str(sublist[0] + 1) for i in range(len(sublist[1]))] if (sublist[0] < len(self.ensembleActivites.copy()) - 1) else ['no teacher' for i in range(len(sublist[1]))] for sublist in enumerate(self.ensembleActivites.copy())]

        # Create a data frame out of the lists
        df = pd.DataFrame({'activity': [item for sublist in Activities for item in sublist],
                           'neuron': [item for sublist in Neuron for item in sublist], \
                           'ensemble': [item for sublist in Ensembles for item in sublist],
                           'trained': [item for sublist in Trained for item in sublist]})

        pd.set_option("display.max_rows", 100)

        # Plot

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.swarmplot(data=df, y="activity", x="ensemble", hue="trained", palette=col, dodge=True, ax=ax, size=8)

        # Connect data points

        for i in range(len(self.ensembleActivites)):
            locs1 = ax.get_children()[i * 2].get_offsets()
            locs2 = ax.get_children()[i * 2 + 1].get_offsets()
            for j in range(locs1.shape[0]):
                x = [locs1[j, 0], locs2[j, 0]]
                y = [locs1[j, 1], locs2[j, 1]]
                ax.plot(x, y, color="black", alpha=0.3)
        ax.set_xlabel('')
        ax.set_ylabel('activity (rate)', fontsize=16)
        ax.legend(title='', fontsize=14, loc='upper right')

        plt.tick_params(axis='both', which='major', labelsize=16)

        plt.show()

    def plotWeightDistribution(self):
        # Extract all trained weights (disregard weights between neurons that are not connected...)
        weights = self.weightPN_PN[-1, self.conPN_PN.astype(bool)].flatten()
        weights0 = self.weightPN_PN[0, self.conPN_PN.astype(bool)].flatten()

        log_weights = np.log(weights)
        log_weights0 = np.log(weights0)

        mymax = max([max(weights0), max(weights)])
        bin_list = np.linspace(0, mymax, num=100)

        mymin_log = min([min(log_weights0), min(log_weights)])
        mymax_log = max([max(log_weights0), max(log_weights)])

        bin_list_log = np.linspace(mymin_log, mymax_log, num=100)

        col = ['#98df8a', '#2ca02c']
        plt.figure(figsize=(12, 8))
        plt.hist(weights0, bins=bin_list, density=True, color='gray', alpha=0.6, label="initialized")
        plt.hist(weights, bins=bin_list, density=True, color=col[1], alpha=0.3, label="trained")
        plt.xlabel('weight', fontsize=16)
        plt.ylabel('probability density', fontsize=16)
        # plt.title('Weight distribution')
        plt.legend(fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.show()

        plt.figure(figsize=(12, 8))
        plt.hist(log_weights0, bins=bin_list_log, density=True, color='gray', alpha=0.6, label="initialized")
        plt.hist(log_weights, bins=bin_list_log, density=True, color=col[1], alpha=0.3, label="trained")
        plt.xlabel('log(weight)', fontsize=16)
        plt.ylabel('probability density', fontsize=16)
        # plt.title('Weight distribution')
        plt.legend(fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.show()

    def plotInput(self):
        time = np.linspace(0, self.time_s, num=self.time_ms)
        with plt.style.context('seaborn-v0_8-dark'):
            # Plot input

            fig = plt.figure()
            for i in range(int(self.time_ms / self.n_burst)):
                plt.hlines(np.where(self.input[:, i * self.n_burst]), i * self.n_burst, i * self.n_burst + self.n_burst,
                           linewidth=1.2)
            plt.title("input")
            plt.xlabel("time (ms)")
            plt.ylabel("input index")
        plt.show()