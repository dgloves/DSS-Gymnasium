"""
This script imports OpenDSSDirect and allows users to load a benchmark IEEE distribution test case system.
Next, you may customize the circuit by adding components:
--> add DERs (solar PV, wind, or BESS) to the system
--> add internal DSS controls to devices and objects
--> add topology reconfigurations
--> add additional real data (loadshapes, DER profiles, etc.)
** Change all template file paths to correct paths on local machine **
"""

from opendssdirect import dss
import pandas as pd
import numpy as np
import os

# set path to import additional time series profile data, profiles, and OpenDSS circuit file(s)
loc_path = os.getcwd()
data_path = r'C:\Users\path\to\time_series_data\data.csv'  # change to correct path
num_steps = 24  # 24 steps in simulation
step_size = 60  # hourly step


def loadcircuit():
    """load desired IEEE circuit from dss file, set basic params"""
    dss.Command('ClearAll')  # clears dss cache
    dss.Command("Redirect 'C:/Users/path/to/openDSS/Circuit_Master_file.dss'")  # change to correct path
    dss.Command('set ControlMode=OFF')  # disable or enable all default controls
    dss.Command('solve')  # get ss power flow of circuit


def importdata():
    """import additional data from csv or text file i.e load curves, PV irradiance/temp data, Wind output, etc"""
    mydata = pd.read_csv(data_path + r'\data.csv')
    mydata = mydata.reset_index(drop=True)  # convert time series idx
    column = 'Output'
    mydata[column] = mydata[column] / mydata[column].abs().max()  # normalize data column
    time_series = mydata[["Output"]].to_numpy()  # convert to numpy array
    return time_series


"""
Add helper functions to build additional circuit components using the imported data, such as:
-->  XY curves
-->  Loadshapes
-->  DERs
-->  Monitors
"""

def buildXYCurves():
    """ Add custom XY curves for temperature, efficiency, volt-VAR control, etc. See DSS manual for further info """
    # PV efficiency curve (all PVs)
    dss.Command('New XYCurve.DER_eff')
    eff_xarr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    eff_yarr = np.array([0.75, 0.78, 0.8, 0.83, 0.86, 0.89, 0.93, 0.95, 0.97, 1.0])
    dss.XYCurves.Npts(10)
    dss.XYCurves.XArray(eff_xarr)
    dss.XYCurves.YArray(eff_yarr)


def buildLoadshape():
    """ add new loadshape to system """
    loadshape1 = pd.read_csv(data_path + r'\Loadshape1.csv', parse_dates=True)
    loadshape_1 = loadshape1.to_numpy()
    # set new loadshape after resampling
    dss.Command('New Loadshape.myloadshape')
    dss.LoadShape.Npts(num_steps)
    dss.LoadShape.MinInterval(step_size)
    dss.LoadShape.PMult(loadshape_1)
    dss.LoadShape.QMult(loadshape_1)


def buildDERs():
    """
    Add PV, Wind, or Storage elements
    No inverter control implemented with THIS PV System.
    Inverters may be assigned to any DER in a separate function for extended control.  See DSS manual for further info.
    """
    dss.Command('New PVSystem.myPV phases=3 bus1=bus_number kV=4.16 kVA=100 irradiance=1 Pmpp=95 conn=delta'
                ' temperature=25 effcurve=DER_eff P-TCurve=myPT Daily=irrad TDaily=myTemp'
                ' %cutin=0.01 %cutout=0.01 kvarMax=44 kvarMaxAbs=44')


def buildMonitors():
    """ add monitors to lines, loads, and circuit elements of choice """
    for load in dss.Loads.AllNames():
        dss.Command('New Monitor.' + load)
        dss.Monitors.Element('Load.' + load)
        dss.Monitors.Terminal(1)  # phase a
        dss.Monitors.Mode(1)  # powers (all phases)
        dss.Command('~ ppolar=no')


def runCircuit():
    loadcircuit()
    buildXYCurves()
    buildLoadshape()
    buildDERs()
    buildMonitors()

if __name__ == '__main__':
    runCircuit()