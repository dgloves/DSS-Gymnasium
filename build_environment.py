"""
This file builds the Gymnasium Environment class around the constructed DSS circuit file build_circuit.py
Follow the guidelines in https://gymnasium.farama.org/introduction/create_custom_env/
"""
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict  # gymnasium spaces
import opendssdirect as dss
import build_circuit
from build_circuit import *  # or alternative globals
import pandas as pd
import csv

class myAgent(gym.Env):
    def __init__(self):
        super().__init__()

        # set output path to write to csv optional
        self.output_path = data_path + r'\gym_env_training_data.csv'

        # dss direct cmds (add if necessary)
        self.Circuit = dss.Circuit
        self.Command = dss.Text.Command
        self.Storage = dss.Storages
        self.Solution = dss.Solution

        # simulation params
        self.num_DERs = len(dss.Storages.AllNames())  # or PVsystems
        self.buses = sorted(self.Circuit.AllBusNames(), key=int)  # bus ID list strings (all buses in network)
        self.Terminated = False
        self.max_step = 24  # fix num steps in sim before reset()
        self.current_step = 1

        """
        Define action and observation spaces as gym.spaces objects based on device controls, ratings, etc.
        These spaces are vectorized and often utilize the underlying NumPy multi-dimensional array structure,
        specific to the Optimization problem at hand.  For example, a single PV System or BESS may have a certain
        nameplate rating, normalized to p.u. or a set of switches may only have a binary action set (on/off)

        observations: any relavent information observable by the agent.  Is the agent mimicking a Distribution System 
        Operator (centralized) with visibility over the entire network, or is the agent local/decentralized with 
        limited visibility (i.e. local DER control vs. centralized dispatch to multiple DERs)?
        """

        # Example actions are charging/discharging of BESS (EXAMPLE)
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.num_DERs,), dtype=np.float64)

        # Example observations of an assumed cost function metric, BESS SoC, and bus voltages (EXAMPLE)
        self.observation_space = Dict(
        {'cost': Box(low=0.0, high=1.0, shape=(self.cost,), dtype=np.float64),
         'soc': Box(low=0.0, high=1.0, shape=(self.num_DERs,), dtype=np.float64),
         'bus_voltage': Box(low=0.9, high=1.1, shape=(self.num_DERs,), dtype=np.float64)})



    def DSSSolutionParams(self):
        """ Set dss Solution params:  https://opendss.epri.com/Solution1.html"""
        self.Command('Set voltagebases=[add voltage bases]')
        self.Command('calc')
        self.Command('Set mode=daily number=number of solves per power flow')
        self.Command('Set hour=0')

    """
    Add helper functions using DSS Direct cmds to apply actions, get observations, etc.
    """

    def Helpers(self):
        """ Add methods to perform environment-related tasks, retrieve info from Monitors, devices, and elements"""
        data = []
        return data


    def Observations(self):
        """
        Build observation vector or dict for agent to match defined space in __int__, import helpers or
        """
        observations = []
        observations = observations.flatten()  # may need to flatten this vector
        return


    def ApplyAction(self, action):
        """
        Apply action to device or element in circuit (i.e. power dispatch update, switch flip, etc.)
        :param action:
        :return: n/a
        """


    def Reward(self, voltages):
        """
        Build reward function based on optimization objective and constraints
        Example: build reward function based on operational voltage limit violations:
        --> voltage at bus within operational limits [0.95,1.05]pu = 0, else penalty
        :param voltages:
        :return: total reward
        """
        v_reward = []
        for voltage in voltages:
            if  0.95 <= voltage <= 1.05:
                volt_reward = 0.0
                v_reward.append(volt_reward)
            else:
                volt_reward = -1.0
                v_reward.append(volt_reward)
        reward = sum(v_reward)
        return reward


    def AdditionalInfo(self, value):
        """
        Any additional info not directly observed by agent which may be useful for decision-making
        :param value: some extra data
        Returns: info dict
        """
        info_dict = {}
        info_dict['key'] = value
        return info_dict


    def step(self, action):
        """
        Apply agent actions at each time step of the simulation, compute load flow, gather observations,
        and compute reward
        :param action:
        :return: observations, reward, done, info
        """
        # print('action:', action)
        self.ApplyAction(action)
        self.Solution.Solve()  # load flow

        # get new state observations
        observation = self.Observations()
        info = self.AdditionalInfo(data)  # feed in data from helpers
        # info = {}  use if no additional info
        reward = self.Reward(voltages)  # custom reward function, feed in voltages from observations
        if self.current_step == self.max_step:
            self.Terminated = True
        else:
            self.Terminated = False
            self.current_step += 1
        return observation, reward, self.Terminated, False, info


    def reset(self, seed=None, options=None):
        """
        This resets the build_circuit.py file and clears the OpenDSS cache to restart a new simulation/episode
        :param seed: starting seed for episode
        :param options: address a circuit parameter for
        :return: circuit steady state observations, info
        """
        print('resetting DSS environment')
        build_circuit.runCircuit()  # reset circuit
        self.DSSSolutionParams()
        self.current_step = 1
        observation = self.Observations()
        info = {}  # add info or none to start sim
        self.Terminated = False
        return observation, info

    def render(self):
        # add only if necessary
        pass

    def close(self):
        # n/a in most cases
        pass


