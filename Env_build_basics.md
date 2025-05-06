# Gymnasium Environment Construction

## Step 1: Building your OpenDSS Circuit
The first step in constructing a gymnasium environment involves importing your own personal benchmark distribution circuit via OpenDSS on which the user intends to conduct a study. 
These circuits are commonly used amongst the power systems research community, and they provide a solid starting point for evaluating your RL algorithms in both a centralized (operator) and/or a decentralized (device) approach.  After downloading OpenDSS, the IEEE benchmark distribution files can usually be found in the "./local_download_path/OpenDSS/IEEETestCases" folder and we have included a few of these circuits in the main directory for convenience.  More importantly, these circuits are only baseline models, and depending on the study, users typically will want to add components, devices, etc. to the circuit to simulate a modern realistic distribution system, including:
 
 * Loads and Generators
 * Distributed Energy Resources (Solar PV Systems, Wind, Battery Energy Storage Systems) (with or without inverter objects)
 * Loadshape/Time Series profiles (PV irradiance/temperature data, loadshape P,Q curves, etc.) for QSTS (quasi-static time-series) simulations
 * Monitoring and metering infrastructure (Monitors, Energy Meters)
 * Additional circuit components (switches, regulators, capacitor banks, etc.)
   
Generally, this can be accomplished in two ways:
1. Create a name.dss file by opening OpenDSS and following a similar structure as seen in the example files or in this [discussion](https://sourceforge.net/p/electricdss/discussion/).  This name.dss file can then be called in the Master.dss file when compiling the circuit as seen in the provided benchmark test system file folders:
 ---
 Add a new load:
 ---
 New Load.S19a Bus1=19.1 Phases=1 Conn=Wye Model=1 kV=2.4 kW=40.0  kvar=20.0
 
 ---
 Add a new PV System:
 ---
 New PVSystem.MyPV phases=3 conn=wye bus1=68 kV=4.8 kVA=100 irrad=1 Pmpp=95 temperature=25 PF=1 effcurve=Myeffcurve P-TCurve=MyPTcurve Daily=Myirradcurve TDaily=Mytempdata





![DSS Example](dss_example.PNG "OpenDSS File Add Load and PV System to Circuit")


The Master.dss file then calls all other name.dss files and sets the circuit up for use.

2. Using the OpenDSSDirect interface, create a .py file which performs similar operations:
   
   '''python
   from opendssdirect import dss
   import pandas as pd
   '''

   Import the desired circuit
   '''python
   dss('Redirect "local_path/Master.dss"')
   '''
 
   
