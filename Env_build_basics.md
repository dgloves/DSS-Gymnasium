# Gymnasium Environment Construction

## Step 1: Building your OpenDSS Circuit
The first step in constructing a gymnasium environment involves importing your own personal benchmark distribution circuit via OpenDSS on which the user intends to conduct a study.  These circuits are commonly used amongst the power systems research community, and they provide a solid starting point for evaluating your RL algorithms in both a centralized and decentralized approach.  After downloading OpenDSS, the IEEE benchmark distribution files can usually be found in the "./local_path/OpenDSS/IEEETestCases" folder and we have included a few of these circuits [here](./)  Generally, this can be accomplished in two ways:

