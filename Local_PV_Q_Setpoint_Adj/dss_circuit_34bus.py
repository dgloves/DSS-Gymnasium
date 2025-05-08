"""
openDSSDirect circuit import
ieee 34bus test system with single PV system
single DER, local voltage regulation via reactive power setpoint manipulation (no droop)
"""

import pandas as pd
import numpy as np
from opendssdirect import dss
import os

# data_path = os.getcwd()  # local dir
dss_path = r'C:\Users\dglov\OneDrive\Desktop\OpenDSS\34Bus\ieee34Mod1.dss'
data_path = r'C:\Users\dglov\OneDrive\Desktop\OpenDSS\34Bus'

num_steps = 8640  # 60 days (5760), 90 days 8640
step_size = 15  # 15 min
Sbase = 1e6
num_pvs = 1


def load34bus():
    dss.Command('ClearAll')
    dss.Command("Redirect 'C:\\Users\\dglov\\OneDrive\\Desktop\\OpenDSS\\34Bus\\ieee34Mod1.dss'")
    dss.Loads.Status(3)  # response to load mult = variable
    dss.Command('set ControlMode=OFF')  # disable voltage regulators for flexibility
    dss.Command('solve')


def importPVData():
    """load hourly PV output time series data from NSRD https://nsrdb.nrel.gov/"""
    pv_data = pd.read_csv(data_path + r'\pv_profile_60min.csv', parse_dates=['LocalTime'], index_col=['LocalTime'])
    pv_data = pv_data.resample("15min").asfreq()
    pv_output = pv_data.interpolate(method='linear')
    pv_output.drop(pv_output.tail(1).index, inplace=True)
    data = pv_output['2006-06-01':'2006-08-29']  # slice 90 days of data June-Aug Central TX
    data = data.reset_index(drop=True)
    df = data.copy()
    column = 'Power(kW)'
    df[column] = df[column] / df[column].abs().max()
    pv_time_series = df[["Power(kW)"]].to_numpy()
    return pv_time_series


def buildXYs():
    # PV For Pmpp at 25 deg celcius max efficiency
    dss.Command('New XYCurve.PV_temp')
    temp_xarr = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    power_yarr = np.array([0.82, 0.92, 0.97, 0.98, 0.99, 1.0, 0.99, 0.97, 0.89, 0.8, 0.75, 0.7, 0.65])
    dss.XYCurves.Npts(13)
    dss.XYCurves.XArray(temp_xarr)
    dss.XYCurves.YArray(power_yarr)

    # PV efficiency curve (all PVs)
    dss.Command('New XYCurve.PV_eff')
    eff_xarr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    eff_yarr = np.array([0.75, 0.78, 0.8, 0.83, 0.86, 0.89, 0.93, 0.95, 0.97, 0.99])
    dss.XYCurves.Npts(10)
    dss.XYCurves.XArray(eff_xarr)
    dss.XYCurves.YArray(eff_yarr)


def resampleDF(df):
    """ resample loadshapes and temperature curves to match time series"""
    df['LocalTime'] = pd.to_datetime(df['LocalTime'])
    df.set_index('LocalTime', inplace=True)
    df = df.resample('15min').asfreq()
    df = df.interpolate(method='linear')
    df = df['2006-06-01':'2006-08-29']
    df = df.reset_index(drop=True)
    return df


def assignLoadShapes():
    """assign random loadshape types: residential, commercial, industrial to all spot loads"""
    count = 1
    for name in dss.Loads.AllNames():
        dss.Loads.Name(name)  # activate load
        if count == 1:
            dss.Loads.Daily('lshape_1')
        elif count == 2:
            dss.Loads.Daily('lshape_2')
        else:
            dss.Loads.Daily('lshape_3')
            # reset counter
        if count > 3:
            count = 1
        else:
            count += 1


def buildLoadshapes(pv_time_series):
    loadshape1 = pd.read_csv(data_path + r'\Loadshape1.csv', parse_dates=True)
    loadshape2 = pd.read_csv(data_path + r'\Loadshape2.csv', parse_dates=True)
    loadshape3 = pd.read_csv(data_path + r'\Loadshape3.csv', parse_dates=True)

    loadshape1_summer = resampleDF(loadshape1)
    loadshape2_summer = resampleDF(loadshape2)
    loadshape3_summer = resampleDF(loadshape3)

    loadshape_1 = loadshape1_summer.to_numpy()
    loadshape_2 = loadshape2_summer.to_numpy()
    loadshape_3 = loadshape3_summer.to_numpy()

    dss.Command('New Loadshape.lshape_1')
    dss.LoadShape.Npts(num_steps)
    dss.LoadShape.MinInterval(step_size)
    dss.LoadShape.PMult(loadshape_1)
    dss.LoadShape.QMult(loadshape_1)

    dss.Command('New Loadshape.lshape_2')
    dss.LoadShape.Npts(num_steps)
    dss.LoadShape.MinInterval(step_size)
    dss.LoadShape.PMult(loadshape_2)
    dss.LoadShape.QMult(loadshape_2)

    dss.Command('New Loadshape.lshape_3')
    dss.LoadShape.Npts(num_steps)
    dss.LoadShape.MinInterval(step_size)
    dss.LoadShape.PMult(loadshape_3)
    dss.LoadShape.QMult(loadshape_3)

    # PV loadshape
    dss.Command('New Loadshape.irrad')
    dss.LoadShape.Npts(num_steps)
    dss.LoadShape.MinInterval(step_size)
    dss.LoadShape.PMult(pv_time_series)


# import weather temp for PV
def buildTempCurves():
    pv_temps_hourly_year = pd.read_csv(data_path + r'\dallas_tx_pv_temp_60min.csv', parse_dates=True)
    pv_temp_summer = resampleDF(pv_temps_hourly_year)
    pv_temp = pv_temp_summer.to_numpy()
    pv_temp = np.transpose(pv_temp).tolist()
    pv_temp = pv_temp[0]
    return pv_temp


def buildPV():  # match load pf, set reactive power limit = 44% * Srated
    """
    No inverter control implemented with PV System.  Agent will access PVSystem directly for Q adjustments.
    Q limit = 44% * S_rated (IEEE 1547)
    """
    dss.Command('New PVSystem.pv890 phases=3 bus1=890 kV=4.16 kVA=550 irradiance=1 Pmpp=500 conn=delta'
                ' temperature=25 effcurve=PV_eff P-TCurve=PV_temp Daily=irrad TDaily=Temp'
                ' %cutin=0.01 %cutout=0.01 kvarMax=242 kvarMaxAbs=242')


def buildMonitors():
    dss.Command('New Monitor.PV_sys_power')
    dss.Monitors.Element('PVSystem.pv890')
    dss.Monitors.Terminal(1)
    dss.Monitors.Mode(1)  # P,Q
    dss.Command('~ ppolar=no')

    dss.Command('New Monitor.Bus890_voltage')
    dss.Monitors.Element('PVSystem.pv890')
    dss.Monitors.Terminal(1)
    dss.Monitors.Mode(0)  # V,I



def run34busCircuit():
    pv_data = importPVData()
    load34bus()
    buildXYs()
    buildLoadshapes(pv_data)
    assignLoadShapes()
    pv_temp = buildTempCurves()
    dss.Command('New Tshape.Temp npts=8640 minterval=15 temp='+str(pv_temp))
    buildPV()
    buildMonitors()


if __name__ == '__main__':
    run34busCircuit()
