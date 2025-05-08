"""
build gymnasium environment class to run ieee34 bus dss circuit 'dss_circuit_34bus.py'
"""

import dss_circuit_34bus
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import opendssdirect as dss
import numpy as np
import pandas as pd
import csv
import os
data_path = os.getcwd()


class LocalPV_Agent(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.output_path = data_path + r'\data\train_agent_DQN.csv'

        # dss direct cmds to subclass (optional)
        self.Bus = dss.Bus
        self.Circuit = dss.Circuit
        self.Command = dss.Command
        self.CktElement = dss.CktElement
        self.Element = dss.Element
        self.Loads = dss.Loads
        self.Loadshape = dss.LoadShape
        self.PVsystems = dss.PVsystems
        self.Solution = dss.Solution

        # set params for 34 bus circuit
        self.mybus = '890'
        self.mypv = 'pv890'
        self.Sbase = 1e6
        self.current_step = 1
        self.max_step = 8640
        self.begin = True
        self.Terminated = False

        # sim limits on voltage, reactive power limits (set on PVSystem)
        self.Vpu_max = 1.05
        self.Vpu_min = 0.95
        self.count = 0
        self.voltage_violation_count = 0
        self.q_violation_count = 0
        self.Qpv_llim = -242.0
        self.Qpv_ulim = 242.0
        # self.PV_kVAR_Setpoint_Start = self.PVsystems.kvar()

        # configure action and observation spaces
        self.actions = np.array([0,1,2])  # 0 = do nothing, 1 = lower kVAR setpoint, 2 = raise kVAR setpoint
        self.num_actions = len(self.actions)
        self.action_space = Discrete(self.num_actions)
        # local voltage measurement at PCC
        self.observation_space = Box(low=0.9, high=1.1, shape=(1,), dtype=np.float64)


    # dss solve params
    def sysFlatStart(self):
        dss_circuit_34bus.run34busCircuit()


    def setSolutionParams(self):
        self.Command('Set voltagebases=[69.0 24.9 4.16 0.48]')
        self.Command('calc')
        self.Command('Set mode=daily number=1')
        self.Solution.StepSizeMin(15)
        self.Command('Set hour=0')


    # observations
    def obsBusV(self):
        self.Circuit.SetActiveBus(self.mybus)
        vpu = self.Bus.PuVoltage()[0]
        return vpu


    def obsPVSysPowers(self):
        self.PVsystems.Name(self.mypv)
        p = self.PVsystems.kW()
        q = self.PVsystems.kvar()
        s = self.PVsystems.kVARated()
        qpu = round((q / s), 3)
        ppu = round((p / s), 3)
        return s, p, q, ppu, qpu  # return PV system powers on PV rating base (NOT SYSTEM BASE!!)


    def get_info(self, p, q):
        return {"real_power": p, "reactive_power": q}


    # Apply actions
    def PVSystemReset(self):
        kvar_setpoint = self.PV_kVAR_Setpoint_Start
        self.Circuit.SetActiveElement('PVSystem.pv890')
        self.PVsystems.kvar(kvar_setpoint)


    def applyAction(self, action):
        vpu = self.Bus.PuVoltage()[0]
        if action == 0:
            pass
        elif action == 1:self.lowerkVAR(vpu)
        else: self.raisekVAR(vpu)


    def lowerkVAR(self, Vpu):
        self.PVsystems.Name(self.mypv)
        current_setpoint = self.PVsystems.kvar()
        kVAR = (abs(Vpu - 1)) * 100
        new_setpoint = current_setpoint - kVAR
        self.PVsystems.kvar(new_setpoint)


    def raisekVAR(self, Vpu):
        self.PVsystems.Name(self.mypv)
        current_setpoint = self.PVsystems.kvar()
        kVAR = (abs(Vpu - 1)) * 100
        new_setpoint = current_setpoint + kVAR
        self.PVsystems.kvar(new_setpoint)


    # reward function(s)
    def checkQNameplate(self, s, p, q):
        """validate available Q_pv """
        qlim = np.sqrt(s**2 - p**2)
        penalty = 0 if abs(q) <= abs(qlim) else -1
        return penalty


    def checkQ1547(self, s, q):
        """validate Q_pv IEEE 1547 limits"""
        qlim = 0.44 * s
        penalty = 0 if abs(q) <= abs(qlim) else -1 * ((q - qlim)**2)
        if penalty < 0:
            self.q_violation_count += 1
        else:
            pass
        return penalty


    def checkBusVoltage(self, bus):
        """validate operational voltage limits"""
        vbus = self.obsBusV()
        dev_penalty = -1 * ((vbus - 1)**2)
        if vbus > 1.05 or vbus < 0.95:
            vlim_penalty = -1
            self.voltage_violation_count += 1
        else:
            vlim_penalty = 0
        penalty = dev_penalty + vlim_penalty
        return penalty


    def reward(self):
        """constraints with penalty-based reward"""
        s, p, q, ppv_pu, qpv_pu = self.obsPVSysPowers()
        nameplate_penalty = self.checkQNameplate(s, p, q)
        stds_penalty = self.checkQ1547(s, q)
        voltage_penalty = self.checkBusVoltage(self.mybus)
        # reward = nameplate_penalty + stds_penalty + voltage_penalty
        reward = voltage_penalty  # voltage reg only
        return reward


    def step(self, action):
        self.applyAction(action)
        self.Solution.Solve()
        self.Solution.FinishTimeStep()
        obs = np.array([self.obsBusV()]).flatten()
        s, p, q, ppv_pu, qpv_pu = self.obsPVSysPowers()
        info = self.get_info(ppv_pu, qpv_pu)  # pv power p.u. to dict
        reward = self.reward()
        if self.current_step == self.max_step:
            self.Terminated = True
        else:
            self.Terminated = False
            self.current_step += 1
        return obs, reward, self.Terminated, False, info  # no truncation


    def reset(self, seed=None, options=None):
        print('Resetting DSS environment')
        self.sysFlatStart()
        self.setSolutionParams()
        obs = np.array([self.obsBusV()]).flatten()
        s, p, q, ppv_pu, qpv_pu = self.obsPVSysPowers()
        info = self.get_info(ppv_pu, qpv_pu)
        self.current_step = 0
        self.Terminated = False
        self.begin = True
        return obs, info


    def render(self):
        # add if necessary
        pass


    def close(self):
        # n/a
        pass