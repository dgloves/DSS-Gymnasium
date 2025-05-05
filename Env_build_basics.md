# Gymnasium Environment Construction

## Step 1: Building your OpenDSS Circuit
The first step in constructing a gymnasium environment involves importing your own personal benchmark distribution circuit via OpenDSS on which the user intends to conduct a study. 
 These circuits are commonly used amongst the power systems research community, and they provide a solid starting point for evaluating your RL algorithms in both a centralized (operator) and/or a decentralized (device) approach.  After downloading OpenDSS, the IEEE benchmark distribution files can usually be found in the "./local_download_path/OpenDSS/IEEETestCases" folder and we have included a few of these circuits in the main directory for convenience.  More importantly, these circuits are only baseline models, and depending on the study, users typically want to add components, devices, etc. to the circuit, including:
 
 * Distributed Energy Resources (Solar PV Systems, Wind, Battery Energy Storage Systems) (with or without inverter control)
 * Loadshape profiles (PV irradiance, load curves, etc.) for QSTS (quasi-static time-series) simulations
 * Monitoring and metering infrastructure
 * Additional circuit components
   
 Generally, this can be accomplished in two ways:

