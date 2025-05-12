"""
D. Glover  Google Colab - Single PV agent IEEE 123bus local PV System voltage deviation minimization
build gymnasium environment class to run dss circuit 'dss_circuit_123bus_singlePV.py'

"""

import dss_circuit_123bus_singlePV
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import opendssdirect as dss
import numpy as np
import os
import random as rd
data_path = os.getcwd()


class SinglePV_Agent(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.output_path = data_path + r'\data\train_agent_singlePV_123bus.csv'  # write to csv during step()

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

        # set params for circuit
        self.mybus = '71'
        self.mypv = 'pv71'
        self.num_pvs = len(self.PVsystems.AllNames())
        self.Sbase = 1e6
        self.current_step = 1
        self.max_step = 2016  # set episodes at 24 hrs x 7 days: 5 min steps
        self.total_steps = 8640  # 30 days
        self.begin = True
        self.Terminated = False

        # sim limits on voltage, reactive power limits (set on PVSystem)
        self.Vpu_max = 1.05
        self.Vpu_min = 0.95
        self.count = 0
        self.voltage_violation_count = 0
        self.q_violation_count = 0
        self.Qpv_llim = -66
        self.Qpv_ulim = 66

        # configure action and observation spaces
        # set action space to 44% kVA nameplate per unit
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        # local voltage measurement at bus 71
        self.observation_space = Box(low=0.9, high=1.1, shape=(1,), dtype=np.float64)


    # dss solve params
    def sysFlatStart(self):
        dss_circuit_123bus_singlePV.run123busCircuit()


    def setSolutionParams(self):
        """ set voltage bases for circuit and apply random seed to episode starting point"""
        self.Command('Set voltagebases=[4.16 0.48]')
        self.Command('calc')
        self.Command('Set mode=daily number=1')
        self.Solution.StepSizeMin(5)
        starting_point = rd.randint(0, self.total_steps - self.max_step)  # randomize starting point
        print('starting_5min_point:', starting_point)
        self.Command('Set hour=' + str(starting_point))
        return starting_point


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
        qpu = round((q / s), 5)
        ppu = round((p / s), 5)
        return s, p, q, ppu, qpu  # return PV system powers on PV rating base (NOT SYSTEM BASE!!)


    def get_info(self, p, q):
        """ add any relavant observable local data - pv71 powers"""
        return {"real_power": p, "reactive_power": q}


    def applyQSetpoint(self, action):
        self.PVsystems.Name(self.mypv)
        s = self.PVsystems.kVARated()
        qpu = action * s  # take pu of nameplate
        self.PVsystems.kvar(qpu)


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


    def checkBusVoltage(self):
        """check for voltage deviation from 1pu + penalty for operational violation"""
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
        """voltage deviation + operational voltage violation + pv_nameplate_check"""
        s, p, q, ppv_pu, qpv_pu = self.obsPVSysPowers()
        nameplate_penalty = self.checkQNameplate(s, p, q)
        stds_penalty = self.checkQ1547(s, q)
        voltage_penalty = self.checkBusVoltage()
        reward = nameplate_penalty + stds_penalty + voltage_penalty
        return reward


    def step(self, action):
        self.applyQSetpoint(action)
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