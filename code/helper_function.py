#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import requests
import json
import logging
import easy_biologic as ebl
import easy_biologic.base_programs as blp
import time
import os
import math
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.signal import savgol_filter
from argparse import Namespace
from dragonfly import load_config_file, multiobjective_maximise_functions, maximise_function, maximise_multifidelity_function
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# IP of OT-2 and Potentiostat
Robot_IP = "169.254.105.36"
Headers = {"opentrons-version": "3"}
Biologic = ebl.BiologicDevice('169.254.144.231')

On = json.dumps({"on": True, "waitUntilComplete": True})
Off = json.dumps({"on": False, "waitUntilComplete": True})

# Potentiostat Channels
Channel = [0]

# Data Saving Directory
Path = 'D:/EC-Lab Data/Kai-Jui/20251026-2/'

# Labware Variables
# Tip_Rack = "opentrons_96_filtertiprack_200ul"
Tip_Rack_1 = "opentrons_96_tiprack_300ul"
Tip_Rack_2 = "opentrons_96_tiprack_300ul"
Tip_Rack_3 = "opentrons_96_tiprack_300ul"
Tip_Rack_4 = "opentrons_96_tiprack_300ul"
Tip_Rack_5 = "opentrons_96_tiprack_300ul"
Tip_Rack_6 = "opentrons_96_tiprack_300ul"
Test_plate = "nest_96_wellplate_200ul_flat"
Mixing_plate = "nest_96_wellplate_200ul_flat"
#Mixing_plate = "nest_96_wellplate_2ml_deep"
Reservior = 'nest_96_wellplate_2ml_deep'
Fan = "opentrons_96_tiprack_300ul"
Electrolytes = 'nest_96_wellplate_2ml_deep'
Pipette_Left = "p300_multi_gen2"
#Pipette_Right = "p300_single_gen2"
Pipette_Right = "p300_single"
Temperature_Module = "temperatureModuleV2"

# Labware Location
Test_Plate_Slot = {"slotName": "7"}
Mixing_Plate_Slot = {"slotName": "2"}
Tip_Rack_1_Slot = {"slotName": "6"}
Tip_Rack_2_Slot = {"slotName": "9"}
Tip_Rack_3_Slot = {"slotName": "11"}
Tip_Rack_4_Slot = {"slotName": "4"}
Tip_Rack_5_Slot = {"slotName": "5"}
Tip_Rack_6_Slot = {"slotName": "8"}
Reservior_Slot = {"slotName": "1"}
Fan_Slot = {"slotName": "10"}
Electrolytes_Slot = {"slotName": "3"}

# Labware Calibrated Location

Mixing_Location = {"origin": "top", "offset": {"x": 0, "y": 1, "z": -9}}
Mixing_Leave_Location = {"origin": "top", "offset": {"x": 0, "y": 1, "z": 0}}

Pick_Up_Tip_Location = {"origin": "top", "offset": {"x": 0, "y": 1, "z": 2}}
Drop_Tip_Location = {"origin": "top", "offset": {"x": 0, "y": 1, "z": -15}}

Electrolyte_Aspirate_Location = {"origin": "top", "offset": {"x": 0, "y": 0, "z": -35}}
Electrolyte_Leave_Location = {"origin": "top", "offset": {"x": 0, "y": 0, "z": -10}}

Dispense_Location = {"origin": "top", "offset": {"x": 0.5, "y": 1, "z": 30}}
Dispense_Leave_Location = {"origin": "top", "offset": {"x": 0.5, "y": 1, "z": 38}}

Fan_Location = {"origin": "top", "offset": {"x": 0, "y": 0, "z": -40}}
Fan_Leave_Location = {"origin": "top", "offset": {"x": 0, "y": 0, "z": 20}}
Electrode_Pick_Up_Location = {"origin": "top", "offset": {"x": 0, "y": 0, "z": 30}}
Electrode_Test_Location = {"origin": "top", "offset": {"x": 0, "y": -1.5, "z": 15}}
Electrode_Test_Leave_Location = {"origin": "top", "offset": {"x": 0, "y": -1.5, "z": 80}}

Electrode_Calibration_Location = {"origin": "top", "offset": {"x": -0.5, "y": -0.5, "z": -20}}
Electrode_Calibration_Leave_Location = {"origin": "top", "offset": {"x": -0.5, "y": -0.5, "z": 0}}

Wash_Location = {"origin": "top", "offset": {"x": 0, "y": 0, "z": -40}}
Wash_Leave_Location = {"origin": "top", "offset": {"x": 0, "y": 0, "z": 20}}


# Samples Variable
room_temperature = 19.6
Samples_Number = 4 # maximum = 23
start_location = 1
#test_location = start_location + 4
test_location = start_location
Electrolytes_Number = 5
electrolyte_tip_number = 96
#tip_number = electrolyte_tip_number - 4 - 8 * (Electrolytes_Number + 1)
tip_number = electrolyte_tip_number - 8 * (Electrolytes_Number + 1)
tip_rack_number = 1
electrolyte_tip_rack_number = tip_rack_number
tip_rack = None
drop_tip_rack = None
electrolyte_start_column = 6
flowrate = [5, 10, 10, 50, 50]
# volume_list = [[16, 163, 18, 0, 53], #HE_ClO4-4
#                [0, 239, 0, 0, 11], #3.82m ClO4
#                [0, 151, 72, 5, 22], #HE_ClO4-4-new
#                [0, 214, 0, 0, 36]] #3.42838m ClO4

volume_list = [[0, 190, 9, 20, 22, 9], #HE_BF4_150mS
               [0, 188, 12, 7, 18, 25], #HE_BF4_112mS
               [0, 202, 0, 0, 0, 48], #8.093712m BF4
               [0, 196, 0, 0, 0, 54]] #7.853688m BF4
               
#[0, 0, 140, 0, 0, 110],
#concentrations = [32.657/250, 20.915/250, 3.575/250, 1.913/250, 0.219/250]
concentration_list = [15/250, 10.915/250, 4.29/250, 4/250, 0.219/250]

# Test Plate Location
Well_Plate_Location = {1: 'A1', 2: 'B1', 3: 'C1', 4: 'D1', 5: 'E1', 6: 'F1', 7: 'G1', 8: 'H1',
                       9: 'A2', 10: 'B2', 11: 'C2', 12: 'D2', 13: 'E2', 14: 'F2', 15: 'G2', 16: 'H2',
                       17: 'A3', 18: 'B3', 19: 'C3', 20: 'D3', 21: 'E3', 22: 'F3', 23: 'G3', 24: 'H3',
                       25: 'A4', 26: 'B4', 27: 'C4', 28: 'D4', 29: 'E4', 30: 'F4', 31: 'G4', 32: 'H4',
                       33: 'A5', 34: 'B5', 35: 'C5', 36: 'D5', 37: 'E5', 38: 'F5', 39: 'G5', 40: 'H5',
                       41: 'A6', 42: 'B6', 43: 'C6', 44: 'D6', 45: 'E6', 46: 'F6', 47: 'G6', 48: 'H6',
                       49: 'A7', 50: 'B7', 51: 'C7', 52: 'D7', 53: 'E7', 54: 'F7', 55: 'G7', 56: 'H7',
                       57: 'A8', 58: 'B8', 59: 'C8', 60: 'D8', 61: 'E8', 62: 'F8', 63: 'G8', 64: 'H8',
                       65: 'A9', 66: 'B9', 67: 'C9', 68: 'D9', 69: 'E9', 70: 'F9', 71: 'G9', 72: 'H9',
                       73: 'A10', 74: 'B10', 75: 'C10', 76: 'D10', 77: 'E10', 78: 'F10', 79: 'G10', 80: 'H10',
                       81: 'A11', 82: 'B11', 83: 'C11', 84: 'D11', 85: 'E11', 86: 'F11', 87: 'G11', 88: 'H11',
                       89: 'A12', 90: 'B12', 91: 'C12', 92: 'D12', 93: 'E12', 94: 'F12', 95: 'G12', 96: 'H12'}

# Tip Rack Location
Tip_Rack_Location = {92: 'A1', 91: 'B1', 90: 'C1', 89: 'D1', 96: 'E1', 95: 'F1', 94: 'G1', 93: 'H1',
                     84: 'A2', 83: 'B2', 82: 'C2', 81: 'D2', 88: 'E2', 87: 'F2', 86: 'G2', 85: 'H2',
                     76: 'A3', 75: 'B3', 74: 'C3', 73: 'D3', 80: 'E3', 79: 'F3', 78: 'G3', 77: 'H3',
                     68: 'A4', 67: 'B4', 66: 'C4', 65: 'D4', 72: 'E4', 71: 'F4', 70: 'G4', 69: 'H4',
                     60: 'A5', 59: 'B5', 58: 'C5', 57: 'D5', 64: 'E5', 63: 'F5', 62: 'G5', 61: 'H5',
                     52: 'A6', 51: 'B6', 50: 'C6', 49: 'D6', 56: 'E6', 55: 'F6', 54: 'G6', 53: 'H6',
                     44: 'A7', 43: 'B7', 42: 'C7', 41: 'D7', 48: 'E7', 47: 'F7', 46: 'G7', 45: 'H7',
                     36: 'A8', 35: 'B8', 34: 'C8', 33: 'D8', 40: 'E8', 39: 'F8', 38: 'G8', 37: 'H8',
                     28: 'A9', 27: 'B9', 26: 'C9', 25: 'D9', 32: 'E9', 31: 'F9', 30: 'G9', 29: 'H9',
                     20: 'A10', 19: 'B10', 18: 'C10', 17: 'D10', 24: 'E10', 23: 'F10', 22: 'G10', 21: 'H10',
                     12: 'A11', 11: 'B11', 10: 'C11', 9: 'D11', 16: 'E11', 15: 'F11', 14: 'G11', 13: 'H11',
                     4: 'A12', 3: 'B12', 2: 'C12', 1: 'D12', 8: 'E12', 7: 'F12', 6: 'G12', 5: 'H12'}

# Electrolyte Location
Electrolytes_Location = {1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'C8', 9: 'C9', 10: 'C10',
                         11: 'C11', 12: 'C12'}


if not os.path.exists(Path + 'Combination.csv'):
    Combination = {'Conductivity': [], 'Potential': [], 'HER': [], 'OER': []}
    for i in range(Electrolytes_Number):
        Combination['Liquid_' + str(i + 1)] = []
        Combination['Liquid_' + str(i + 1) + '_Concentration'] = []
    Combination['Water'] = []
    Combination['Zn_Concentration'] = []
    Comb = pd.DataFrame(Combination)
    print("Created Combination.csv")
else:
    print("Loading Combination.csv")
    Comb = pd.read_csv(Path + 'Combination.csv')
    Combination = Comb.to_dict(orient='list')

if not os.path.exists(Path + 'Summary.csv'):
    Summary = {'Conductivity_Mean': [], 'Conductivity_STD': [], 'Potential_Mean': [], 'Potential_STD': [],
               'OER_Mean': [], 'OER_STD': [], 'HER_Mean': [], 'HER_STD': []}
    for i in range(Electrolytes_Number):
        Summary['Liquid_'+str(i+1)] = []
        Summary['Liquid_' + str(i + 1) + '_Concentration'] = []
    Summary['Water'] = []
    Summary['Zn_Concentration'] = []
    for i in range(4):
        Summary['Conductivity_'+str(i+1)] = []
        Summary['Potential_'+str(i+1)] = []
        Summary['OER_'+str(i+1)] = []
        Summary['HER_' + str(i + 1)] = []
    Sum = pd.DataFrame(Summary)
    print("Created Summary.csv")
else:
    Sum = pd.read_csv(Path + 'Summary.csv')
    Summary = Sum.to_dict(orient='list')
    print("Loading Summary.csv")

"""
OT2 robot helper function
"""

def create_run(robot_ip):
    """
    Create a robot run task.

    Args:
         robot_ip (str): the IP address of the OT2 robot
    """
    runs_url = f"http://{robot_ip}:31950/runs"
    print(f"Command:\n{runs_url}")

    r = requests.post(
        url=runs_url,
        params={"waitUntilComplete": True},
        headers=Headers
    )

    r_dict = json.loads(r.text)
    run_id = r_dict["data"]["id"]
    print(f"Run ID:\n{run_id}")
    return runs_url, run_id

def light(status, url):
    """
    Set light status.

    Args:
        status (str): ON or OFF
        url (str): command url
    """
    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=status)  # ON or OFF

    print(f"Request status:\n{r}\n{r.text}")

def load_module(slot, module, url):
    """
    Load OT2 module

    Args:
        slot (str): the slot number for placing the module
        module (string): the module load name
        url: command url
        """
    command_dict = {
        "data": {
            "commandType": "loadModule",
            "params": {
                "model": module,
                "location": slot
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}")

    r = requests.post(
        url=url,
        headers=Headers,
        params={"waitUntilComplete": True},
        data=command_payload
    )

    r_dict = json.loads(r.text)
    module_id = r_dict["data"]["result"]["moduleId"]
    return module_id

def load_labware(slot, labware, brand, url):
    """
    Load OT2 labware

    Args:
        slot (dict): the slot number for placing the labware
        labware (string): the labware load name
        brand (string): 'opentrons' or 'custom_beta'
        url: command url
    """
    command_dict = {
        "data": {
            "commandType": "loadLabware",
            "params": {
                "location": slot,
                "loadName": labware,
                "namespace": brand,
                "version": 1
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}")

    r = requests.post(
        url=url,
        headers=Headers,
        params={"waitUntilComplete": True},
        data=command_payload
    )

    r_dict = json.loads(r.text)
    labware_id = r_dict["data"]["result"]["labwareId"]
    return labware_id

def load_pipette(pipette, mount, url):
    """
    Load OT2 pipette

    Args:
        pipette (string): the pipette load name
        mount (string): mounting position, 'left' or 'right'
        url: command url
    """
    command_dict = {
        "data": {
            "commandType": "loadPipette",
            "params": {
                "pipetteName": pipette,
                "mount": mount
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}")

    r = requests.post(
        url=url,
        headers=Headers,
        params={"waitUntilComplete": True},
        data=command_payload
    )

    r_dict = json.loads(r.text)
    pipette_id = r_dict["data"]["result"]["pipetteId"]
    return pipette_id

def set_temperature(id, temperature, url):
    """
    Set the temperature of the temperature module

    Args:
        id (str): the ID of temperature module
        temperature (float): the target temperature
        url (str): command url
    """
    command_dict = {
        "data": {
            "commandType": "temperatureModule/setTargetTemperature",
            "params": {
                "moduleId": id,
                "celsius": temperature
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def deactivate_temperature_module(id, url):
    """
    deactivate the temperature module

    Args:
        id (str): the ID of the temperature module
        url (str): command url
    """
    command_dict = {
        "data": {
            "commandType": "temperatureModule/deactivate",
            "params": {
                "moduleId": id,
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def home_robot(url):
    """
    Home the OT2 robot, including robotic arm and pipettes.

    Args:
        url (str): home url
    """
    command_dict = {"target": "robot"}
    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def pick_up_tip(rack, well, location, pipette, url):
    """
    Pick up pipette tip.

    Args:
        rack (string): the labware ID of tip rack
        well (string): the well to pick up the tip
        location (dict): the location to pick up the tip
            Default is {"origin": "top", "offset": {"x": 0, "y": 0, "z": 0}}
        pipette (string): the ID of the pipette that is used to pick up the tip
        url (string): command url
    """
    command_dict = {
        "data": {
            "commandType": "pickUpTip",
            "params": {
                "labwareId": rack,
                "wellName": well,
                "wellLocation": location,
                "pipetteId": pipette
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def drop_tip(rack, well, location, pipette, url):
    """
    Drop pipette tip

    Args:
        rack (string): the labware ID of tip rack or trash bin
        well (string): the well to drop the tip
        location (dict): the location to drop the tip
            Default is {"origin": "top", "offset": {"x": 0, "y": 0, "z": 0}}
        pipette (string): the ID of the pipette that is used to drop the tip
    """
    command_dict = {
        "data": {
            "commandType": "dropTip",
            "params": {
                "labwareId": rack,
                "wellName": well,
                "wellLocation": location,
                "pipetteId": pipette
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def aspirate(labware, well, location, flowrate, volume, pipette, url):
    """
    Aspirate liquid.

    Args:
        labware (string): the ID of the labwere for liquid aspiration
        well (string): the well for liquid aspiration
        location (dict): the location for liquid aspiration
        flowrate (int): flow rate for liquid aspiration
        volume (float): the volume of the aspirating liquid
        pipette (string): the ID of the pipette that is used for aspiration
        url (string): command url
    """
    command_dict = {
        "data": {
            "commandType": "aspirate",
            "params": {
                "labwareId": labware,
                "wellName": well,
                "wellLocation": location,
                "flowRate": flowrate,
                "volume": volume,
                "pipetteId": pipette
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def touch_tip(labware, well, location, speed, pipette, url):
    """
    Touch pipette tip on the labware.

    Args:
        labware (str): the ID of the labware
        well (str): the well for tip touching
        location (dict): the location for tip touching
        speed (float): speed for tip touching
        pipette (str): the ID of the pipette that is used for tip touching
        url (str): command url
    """
    command_dict = {
        "data": {
            "commandType": "touchTip",
            "params": {
                "labwareId": labware,
                "wellName": well,
                "wellLocation": location,
                "speed": speed,
                "pipetteId": pipette
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def dispense(labware, well, location, flowrate, volume, pipette, url):
    """
    Dispense liquid.

    Args:
        labware (string): the ID of the labwere to dispense liquid
        well (string): the well to dispense liquid
        location (dict): the location to dispense liquid
        flowrate (int): flow rate for liquid dispense
        volume (float): the volume of the dispensing liquid
        pipette (string): the ID of the pipette that is used for dispensing
        url (string): command url
    """
    command_dict = {
        "data": {
            "commandType": "dispense",
            "params": {
                "labwareId": labware,
                "wellName": well,
                "wellLocation": location,
                "flowRate": flowrate,
                "volume": volume,
                "pipetteId": pipette
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def blowout(labware, well, location, flowrate, pipette, url):
    """
    Blow out the liquid in the tip

    Args:
        labware (string): the ID of the labwere to blow out liquid
        well (string): the well to blow out liquid
        location (dict): the location to blow out liquid
        flowrate (int): flow rate for blowing out liquid
        pipette (string): the ID of the pipette that is used for dispensing
        url (string): command url
    """
    command_dict = {
        "data": {
            "commandType": "blowout",
            "params": {
                "labwareId": labware,
                "wellName": well,
                "wellLocation": location,
                "flowRate": flowrate,
                "pipetteId": pipette
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def drop_tip_to_trash(pipette, url):
    """
    Drop the used tip to trash bin.

    Args:
        pipette (str): the ID of the pipette that is used to drop the tip
        url (str): command url
    """
    command_dict = {
        "data": {
            "commandType": "moveToAddressableAreaForDropTip",
            "params": {
                "pipetteId": pipette,
                "addressableAreaName": "fixedTrash",
                "offset": {"x": 0, "y": -20, "z": 10}
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")
    
    command_dict = {
        "data": {
            "commandType": "dropTipInPlace",
            "params": {
                "pipetteId": pipette
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def move_to_well(labware, well, location, pipette, url, speed=400):
    """
    Move the pipette to a well.

    Args:
        labware (string): the ID of the labware for the pipette to move to
        well (string): the well for the pipette to move to
        location (dict): the location for the tip to move to
        pipette (string): the ID of the pipette that is used to move to the well
        url (string): command url
        speed (float): the speed of the pipette movement
    """
    command_dict = {
        "data": {
            "commandType": "moveToWell",
            "params": {
                "labwareId": labware,
                "wellName": well,
                "wellLocation": location,
                "speed": speed,
                "pipetteId": pipette
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")

def move_relative(axis, distance, pipette, url, speed=400):
    """
    Move the pipette relative to its current position.

    Args:
        axis (str): direction to move the pipette. 'x', 'y', 'z'
        distance (float): the distance to move the pipette
        pipette (string): the ID of the pipette that is used to move to the well
        url (string): command url
        speed (float): the speed of the pipette movement
    """
    command_dict = {
        "data": {
            "commandType": "moveRelative",
            "params": {
                "axis": axis,
                "distance": distance,
                "speed": speed,
                "pipetteId": pipette
            },
            "intent": "setup"
        }
    }

    command_payload = json.dumps(command_dict)
    print(f"Command:\n{command_payload}\n")

    r = requests.post(
        url=url,
        params={"waitUntilComplete": True},
        headers=Headers,
        data=command_payload
    )

    print(f"Response:\n{r}\n{r.text}\n")


def check_tip():
    global tip_number, tip_rack_number, tip_rack, drop_tip_rack

    tip_number = tip_number - 4
    if tip_number < 1:
        tip_number = 96
        tip_rack_number = tip_rack_number + 1

    if tip_rack_number == 1:
        tip_rack = Labware_3_ID
        drop_tip_rack = Labware_6_ID
    elif tip_rack_number == 2:
        tip_rack = Labware_4_ID
        drop_tip_rack = Labware_7_ID
    elif tip_rack_number == 3:
        tip_rack = Labware_5_ID
        drop_tip_rack = Labware_8_ID
    elif tip_rack_number > 3:
        tip_rack = Labware_3_ID
        drop_tip_rack = Labware_6_ID
        tip_rack_number = 1


def impedance(test, frequency):
    # Run OCP test
    save_path = Path + Well_Plate_Location[test] + '_OCV.csv'

    params_ocv = {
        'time': 2,
        'time_interval': 1,
        # 'voltage_interval': 0.01
    }

    ocv = blp.OCV(
        Biologic,
        params_ocv,
        channels=Channel
    )

    ocv.run('data')
    ocv.save_data(save_path)

    voc = {
        ch: [datum.voltage for datum in data]
        for ch, data in ocv.data.items()
    }

    voc = {
        ch: sum(ch_voc) / len(ch_voc)
        for ch, ch_voc in voc.items()
    }

    # Run PEIS test
    save_path = Path + Well_Plate_Location[test] + '_PEIS.csv'
    
    
    params_peis = {
        'voltage': list(voc.values())[0],
        'final_frequency': frequency,  # frequency unit: Hertz
        'initial_frequency': 1000000,  # frequency unit: Hertz
        'amplitude_voltage': 0.1,  # voltage unit: Volt
        'frequency_number': np.log10(1000000/frequency)*20,
        'duration': 0,  # time unit: second
        # 'vs_final': False,
        # 'time_interval': 1,  #time unit: second
        # 'current_interval': 0.001,
        # 'sweep': 'log',
        'repeat': 10,
        # 'correction': False
        'wait': 0.1
    }

    peis = blp.PEIS(
        Biologic,
        params_peis,
        channels=Channel
    )

    peis.run('data')
    peis.save_data(save_path)


def lsv(test, ocv, end_potential):
    
    start_potential = ocv

    # Run LSV test
    if end_potential > 0:
        save_path = Path + Well_Plate_Location[test] + '_LSV_OER.csv'
        if end_potential > 2.7:
            start_potential = end_potential - 1.5
    elif end_potential < 0:
        start_potential = end_potential + 1.2
        save_path = Path + Well_Plate_Location[test] + '_LSV_HER.csv'

    params_lsv = {
        'start': start_potential,
        'end': end_potential,
        'E2': end_potential,
        'Ef': end_potential,
        # 'vs_initial': [True, False, False, True, True],
        'rate': 0.01,  # unit: V/s. We can change the unit on base_programs if we want.
        'step': 0.001,  # step=dEN/1000
        'N_Cycles': 0,
        'average_over_dE': False,
        'begin_measuring_I': 0.5,
        'End_measuring_I': 1.0
    }

    lsv = blp.CV(
        Biologic,
        params_lsv,
        channels=Channel
    )

    print("Running LSV for " + Well_Plate_Location[test] + " to " + str(end_potential) + " V")

    lsv.run('data')
    lsv.save_data(save_path)


def ca(test):
    # Run CA test
    save_path = Path + Well_Plate_Location[test] + '_CA.csv'

    params_ca = {
        'voltages': [0.5],  # List of voltages in Volts.
        'duration': [30]  # List of times in seconds.
    }

    ca = blp.CA(
        Biologic,
        params_ca,
        channels=Channel
    )

    ca.run('data')
    ca.save_data(save_path)


def cp(test, current):
    # Run CP test
    if current[0] > 0:
        save_path = Path + Well_Plate_Location[test] + '_CP_OER.csv'
    elif current[0] < 0:
        save_path = Path + Well_Plate_Location[test] + '_CP_HER.csv'

    params_cp = {
        'currents': current, #List of currents in Amps.
        'durations': [10, 7, 7, 7] #List of times in seconds.
    }

    cp = blp.CP(
        Biologic,
        params_cp,
        channels=Channel
    )

    cp.run('data')
    cp.save_data(save_path)


def wash(reservoir, pipette, mode, url):
    if mode == 'test':
        # Washed by 1.5M HCl acid (First Column)
        move_to_well(labware=reservoir, well='E2', location=Wash_Leave_Location, pipette=pipette, url=url)
        move_to_well(labware=reservoir, well='E2', location=Wash_Location, pipette=pipette, url=url)
        move_relative(axis="y", distance=10, pipette=pipette, url=url)
        move_relative(axis="y", distance=-20, pipette=pipette, url=url)
        move_relative(axis="y", distance=10, pipette=pipette, url=url)
        time.sleep(5)
        move_to_well(labware=reservoir, well='E2', location=Wash_Leave_Location, pipette=pipette, url=url)
        move_to_well(labware=reservoir, well='E2', location=Wash_Location, pipette=pipette, url=url)
        move_relative(axis="y", distance=10, pipette=pipette, url=url)
        move_relative(axis="y", distance=-20, pipette=pipette, url=url)
        move_relative(axis="y", distance=10, pipette=pipette, url=url)
        time.sleep(5)
        move_to_well(labware=reservoir, well='E2', location=Wash_Leave_Location, pipette=pipette, url=url)
        

    # Washed by DI_water (Second Column)
    move_to_well(labware=reservoir, well='E5', location=Wash_Leave_Location, pipette=pipette, url=url)
    move_to_well(labware=reservoir, well='E5', location=Wash_Location, pipette=pipette, url=url)
    move_relative(axis="y", distance=10, pipette=pipette, url=url)
    move_relative(axis="y", distance=-20, pipette=pipette, url=url)
    move_relative(axis="y", distance=10, pipette=pipette, url=url)
    time.sleep(5)
    move_to_well(labware=reservoir, well='E5', location=Wash_Leave_Location, pipette=pipette, url=url)

    # Washed by Ethanol (Fourth Column)
    move_to_well(labware=reservoir, well='E11', location=Wash_Leave_Location, pipette=pipette, url=url)
    move_to_well(labware=reservoir, well='E11', location=Wash_Location, pipette=pipette, url=url)
    move_relative(axis="y", distance=10, pipette=pipette, url=url)
    move_relative(axis="y", distance=-20, pipette=pipette, url=url)
    move_relative(axis="y", distance=10, pipette=pipette, url=url)
    time.sleep(5)
    move_to_well(labware=reservoir, well='E11', location=Wash_Leave_Location, pipette=pipette, url=url)


def data_process_peis(location):
    print("Processing PEIS for " + location)

    peis_filename = Path + location + '_PEIS.csv'
    # Load data
    #df_peis = pd.read_csv(peis_filename, sep=',', skiprows=1)
    df_peis = pd.read_csv(peis_filename, sep=',')
    df_peis['|Z| [Ohm]'] = df_peis['Impedance modulus']
    #df_peis['|Z| [Ohm]'] = df_peis['abs( Voltage ) [V]'] / df_peis['abs( Current ) [A]']
    df_peis['ReZ [Ohm]'] = df_peis['|Z| [Ohm]'] * np.cos(df_peis['Impedance phase'])
    df_peis['-ImZ [Ohm]'] = df_peis['|Z| [Ohm]'] * (-np.sin(df_peis['Impedance phase']))
    degree = [None] * df_peis.shape[0]
    slope = [None] * df_peis.shape[0]
    for i in range(df_peis.shape[0]):
        if i == 0:
            degree[i] = 0
        if i != 0:
            slope[i] = (df_peis['-ImZ [Ohm]'][i] - df_peis['-ImZ [Ohm]'][i - 1]) / (
                    df_peis['ReZ [Ohm]'][i] - df_peis['ReZ [Ohm]'][i - 1])
            degree[i] = np.arctan(slope[i]) * 180 / math.pi
    df_peis['degree'] = degree
    
    try:
        # Find the intercept
        df_peis_range = pd.DataFrame()
        for i in range(df_peis.shape[0]):
            if i != 0 and df_peis['ReZ [Ohm]'][i - 1] > 0 and df_peis['-ImZ [Ohm]'][i] > 0 > df_peis['-ImZ [Ohm]'][i - 1]:
                df_peis_range = df_peis[(df_peis.index >= (i - 1)) & (df_peis.index <= i)]

        peis_slope, peis_intercept, _, _, _ = linregress(df_peis_range['ReZ [Ohm]'], df_peis_range['-ImZ [Ohm]'])
        re = -peis_intercept / peis_slope
        con = 1000 / re
    except:
        try:
            # Fitting slope
            cursor = 0
            df_peis_range = pd.DataFrame()
            for i in range(df_peis.shape[0]):
                if i != 0 and df_peis['degree'][i] > 44.5 > df_peis['degree'][i - 1]:
                    cursor = i
                if cursor != 0 and df_peis['degree'][i] < 44.5:
                    if (i - cursor) >= 5:
                        df_peis_range = df_peis[(df_peis.index <= (i - 1)) & (df_peis.index >= (cursor - 1))]
                        cursor = 0
                    else:
                        cursor = 0
                elif cursor != 0 and i == (df_peis.shape[0] - 1):
                    if (i - cursor) >= 5:
                        df_peis_range = df_peis[(df_peis.index <= (i - 1)) & (df_peis.index >= (cursor - 1))]
                        cursor = 0
                    else:
                        cursor = 0

            peis_slope, peis_intercept, _, _, _ = linregress(df_peis_range['ReZ [Ohm]'], df_peis_range['-ImZ [Ohm]'])
            re = -peis_intercept / peis_slope
            con = 1000 / re
        except:
            re = 0
            con = 0
    
    peis_data = {'Frequency [Hz]': df_peis['Frequency [Hz]'],
                 '|Z| [Ohm]': df_peis['|Z| [Ohm]'],
                 'ReZ [Ohm]': df_peis['ReZ [Ohm]'],
                 '-ImZ [Ohm]': df_peis['-ImZ [Ohm]'],
                 'degree': df_peis['degree'],
                 }
    peis_data_output = pd.DataFrame(peis_data)
    peis_filename = Path + location + '_PEIS_Calculated.csv'
    peis_data_output.to_csv(peis_filename, index=False)

    if re != 0:
        plt.plot(df_peis['ReZ [Ohm]'], df_peis['-ImZ [Ohm]'], 'ko',
                 df_peis_range['ReZ [Ohm]'], ((df_peis_range['ReZ [Ohm]'] * peis_slope) + peis_intercept), '--r')
        plt.xlabel('ReZ [Ohm]')
        plt.ylabel('-ImZ [Ohm]')
        plt.title(location + '_PEIS')
        plt.savefig(Path + location + '_PEIS.svg')
        plt.close()
    else:
        plt.plot(df_peis['ReZ [Ohm]'], df_peis['-ImZ [Ohm]'], 'ko')
        plt.xlabel('ReZ [Ohm]')
        plt.ylabel('-ImZ [Ohm]')
        plt.title(location + '_PEIS')
        plt.savefig(Path + location + '_PEIS.svg')
        plt.close()

    return re, con


def data_process_lsv(location, re, ocv, mode, end_potential):
    print("Processing LSV for " + location)

    # Determine resistance
    if mode == 'OER':
        lsv_filename = Path + location + '_LSV_OER.csv'
    elif mode == 'HER':
        lsv_filename = Path + location + '_LSV_HER.csv'

    df = pd.read_csv(lsv_filename, sep=',')
    df.drop(df.index[:60], inplace=True)

    try:
        # iR correction
        df['Voltage_Corrected [V]'] = df['Voltage'] - df['Current'] * re * 0.1
        df['Current'] = savgol_filter(df['Current'], window_length=100, polyorder=2)

        potential = pd.DataFrame()

        start_potential = ocv

        if end_potential > 0:
            if end_potential > 2.7:
                start_potential = end_potential - 1.5
        elif end_potential < 0:
            start_potential = end_potential + 1.2
        
        # Find derivative point
        if mode == 'HER':
            potential = (df[(df['Voltage_Corrected [V]'] < start_potential) & (
                    df['Voltage_Corrected [V]'] > end_potential)].reset_index())
            threshold = 2

        elif mode == 'OER':
            potential = (df[(df['Voltage_Corrected [V]'] < end_potential) & (
                    df['Voltage_Corrected [V]'] > start_potential)].reset_index())
            threshold = 1

        current_derivative = np.gradient(potential['Current'], potential['Voltage_Corrected [V]'])
        potential['derivative'] = current_derivative

        # Find the onset potential by locating the index where the derivative surpasses the threshold
        onset_index = np.argmax(potential['derivative'] > np.mean(potential['derivative']) * threshold)
        onset_potential = potential['Voltage_Corrected [V]'][onset_index]

        plt.plot(potential['Voltage_Corrected [V]'], potential['Current'], label='LSV curve')
        plt.axvline(onset_potential, color='r', linestyle='--', label=f'Onset potential: {onset_potential:.5f} V')
        plt.xlabel('Potential (V)')
        plt.ylabel('Current (A)')
        plt.title('LSV with Onset Potential')
        plt.legend()
        plt.grid()

        if mode == 'OER':
            plt.savefig(Path + location + '_LSV_OER.svg')
            plt.close()
        elif mode == 'HER':
            plt.savefig(Path + location + '_LSV_HER.svg')
            plt.close()
    
    except:
        df['Voltage_Corrected [V]'] = df['Voltage'] - df['Current'] * re * 0.1
        onset_potential = 0

    LSV_data = {'Voltage [V]': df['Voltage'],
               'Voltage_Corrected [V]': df['Voltage_Corrected [V]'],
               'Current [A]': df['Current']
               }

    data_output = pd.DataFrame(LSV_data)

    if mode == 'OER':
        lsv_filename_corrected = Path + location + '_LSV_OER_Corrected.csv'
    elif mode == 'HER':
        lsv_filename_corrected = Path + location + '_LSV_HER_Corrected.csv'
        
    data_output.to_csv(lsv_filename_corrected, index=False)
    
    try:
        end_current = df['Current'].iloc[-1]
    except:
        end_current = 0

    return onset_potential, end_current

def data_process_ocv(location):
    print("Processing OCV for " + location)

    ocv_filename = Path + location + '_OCV.csv'
    df_ocv = pd.read_csv(ocv_filename, sep=',')
    ocv = df_ocv['Voltage [V]'].mean()

    return ocv

def data_process_cp(location, re, mode):

    if mode == 'OER':
        cp_filename = Path + location + '_CP_OER.csv'
    elif mode == 'HER':
        cp_filename = Path + location + '_CP_HER.csv'

    df = pd.read_csv(cp_filename, sep=',')

    # iR correction
    df['Voltage_Corrected [V]'] = df['Voltage [V]'] - df['Current [A]'] * re * 0.2

    try:
        potential_1 = df[(df['Time [s]'] >= 15) & (df['Time [s]'] <= 17)].reset_index(drop=True)
        potential_2 = df[(df['Time [s]'] >= 22) & (df['Time [s]'] <= 24)].reset_index(drop=True)
        potential = pd.concat([potential_1, potential_2], axis=0).reset_index(drop=True)
        slope, intercept, _, _, _ = linregress(potential['Voltage_Corrected [V]'], potential['Current [A]'])
        onset_potential = -intercept / slope
    except:
        onset_potential = 0

    CP_data = {'Voltage [V]': df['Voltage [V]'],
                'Voltage_Corrected [V]': df['Voltage_Corrected [V]'],
                'Current [A]': df['Current [A]']
               }
    data_output = pd.DataFrame(CP_data)
    if mode == 'OER':
        cp_filename_corrected = Path + location + '_CP_OER_Corrected.csv'
    elif mode == 'HER':
        cp_filename_corrected = Path + location + '_CP_HER_Corrected.csv'
    data_output.to_csv(cp_filename_corrected, index=False)

    return onset_potential


def calibration(pipette_left, pipette_right, electrolytes, mixing_plate, reservoir, fan, url):
    global tip_number, tip_rack

    # Calibration curve
    Calibration = {'Temperature': [room_temperature], 'Conductivity_1': [], 'Conductance_1': [],
                   'Conductivity_2': [], 'Conductance_2': [], 'Conductivity_3': [], 'Conductance_3': [],
                   'Slope': [], 'Intercept': []}

    standard_conductances = []
    
    calibration_location = [start_location, start_location+1, start_location+2, start_location+3]
    for i in calibration_location:
        # Move to test plate
        move_to_well(labware=mixing_plate, well=Well_Plate_Location[i], location=Electrode_Calibration_Leave_Location,
                     pipette=pipette_right, url=url)
        move_to_well(labware=mixing_plate, well=Well_Plate_Location[i], location=Electrode_Calibration_Location,
                     pipette=pipette_right, url=url)

        # Electrochemical Testing
        impedance(test=i, frequency = 100)
        move_to_well(labware=mixing_plate, well=Well_Plate_Location[i], location=Electrode_Calibration_Leave_Location,
                     pipette=pipette_right, url=url)

        # Washed by DI_Water and ethanol
        wash(reservoir=reservoir, pipette=pipette_right, mode='calibration', url=url)

        # Dried by Fan
        move_to_well(labware=fan, well='G5', location=Fan_Leave_Location, pipette=pipette_right, url=url)
        move_to_well(labware=fan, well='G5', location=Fan_Location, pipette=pipette_right, url=url)

        if i > start_location:
            _, standard_conductance = data_process_peis(location=Well_Plate_Location[i])
            standard_conductances.append(standard_conductance)

            number = i-start_location
            Calibration['Conductance_'+str(number)] = standard_conductance
        
        time.sleep(90)
        move_to_well(labware=fan, well='G5', location=Fan_Leave_Location, pipette=pipette_right, url=url)

    standard_conductivity_25 = np.array([0.5, 12.88, 199.493])
    standard_conductivity_slope = np.array([1.94 / 100, 1.878881988 / 100, 1.878862917 / 100])
    standard_conductivity = standard_conductivity_25 * (1 + (standard_conductivity_slope * (room_temperature - 25)))
    for i in range(3):
        Calibration['Conductivity_' + str(i+1)] = standard_conductivity[i]

    log10_conductivity = np.log10(standard_conductivity)
    log10_conductance = np.log10(standard_conductances)
    log_slope, log_intercept, _, _, _ = linregress(log10_conductance, log10_conductivity)
    Calibration['Slope'] = log_slope
    Calibration['Intercept'] = log_intercept

    # Save calibration curve to file
    Cali = pd.DataFrame(Calibration)
    filename = Path + 'Calibration.csv'
    Cali.to_csv(filename, index=False)

    return log_slope, log_intercept


def to_conductivity(conductance, slope, intercept):
    if conductance <= 0:
        conductivity = 0
    else:
        conductivity = 10**((np.log10(conductance) * slope) + intercept)
    return conductivity


def formulation(combination, pipette_left, electrolytes, water, test_plate, mixing_well, mixing_plate, flowrate, url):
    global electrolyte_tip_rack_number, electrolyte_tip_number, tip_number, tip_rack, drop_tip_rack
    electrolyte_tip_rack = None
    if electrolyte_tip_rack_number == 1:
        electrolyte_tip_rack = Labware_3_ID
    elif electrolyte_tip_rack_number == 2:
        electrolyte_tip_rack = Labware_4_ID
    elif electrolyte_tip_rack_number == 3:
        electrolyte_tip_rack = Labware_5_ID

    # Pick up tip
    pick_up_tip(rack=electrolyte_tip_rack, well=Tip_Rack_Location[electrolyte_tip_number], location=Pick_Up_Tip_Location,
                pipette=pipette_left, url=url)

    # Aspirate Water
    aspirate(labware=electrolytes, well=Electrolytes_Location[water], location=Electrolyte_Aspirate_Location,
             flowrate=50, volume=combination[Electrolytes_Number], pipette=pipette_left, url=url)
    
    time.sleep(1)

    # Dispense Water
    dispense(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location,
             flowrate=50, volume=combination[Electrolytes_Number], pipette=pipette_left, url=url)

    # Blowout Water
    blowout(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location,
            flowrate=50, pipette=pipette_left, url=url)

    # Touch Tip
    touch_tip(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location, speed=10,
              pipette=pipette_left, url=url)

    drop_tip(rack=electrolyte_tip_rack, well=Tip_Rack_Location[electrolyte_tip_number],
            location=Drop_Tip_Location, pipette=pipette_left, url=url)

    time.sleep(1)

    # Add Electrolytes
    for i in range(Electrolytes_Number):
        if combination[i] > 0 and i < 2:
            # Pick up tip
            pick_up_tip(rack=electrolyte_tip_rack, well=Tip_Rack_Location[(electrolyte_tip_number - (8 * (i + 1)))],
                     location=Pick_Up_Tip_Location, pipette=pipette_left, url=url)

            # Aspirate
            aspirate(labware=electrolytes, well=Electrolytes_Location[i + electrolyte_start_column], location=Electrolyte_Leave_Location,
                     flowrate=150, volume=5, pipette=pipette_left, url=url)
                 
            move_to_well(labware=electrolytes, well=Electrolytes_Location[i + electrolyte_start_column], location=Electrolyte_Aspirate_Location,
                         speed=10, pipette=pipette_left, url=url)
        
            aspirate(labware=electrolytes, well=Electrolytes_Location[i + electrolyte_start_column], location=Electrolyte_Aspirate_Location,
                     flowrate=flowrate[i], volume=combination[i], pipette=pipette_left, url=url)
        
            time.sleep(5)

            move_to_well(labware=electrolytes, well=Electrolytes_Location[i + electrolyte_start_column], location=Electrolyte_Leave_Location, speed=1,
                         pipette=pipette_left, url=url)

            time.sleep(5)

            # Dispense
            dispense(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location,
                     flowrate=flowrate[i], volume=combination[i]+5, pipette=pipette_left, url=url)

            # Blowout
            blowout(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location,
                    flowrate=flowrate[i], pipette=pipette_left, url=url)
        
            # Touch Tip
            touch_tip(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location, speed=10,
                      pipette=pipette_left, url=url)

            time.sleep(5)

            drop_tip(rack=electrolyte_tip_rack, well=Tip_Rack_Location[(electrolyte_tip_number - (8 * (i + 1)))],
                    location=Drop_Tip_Location, pipette=pipette_left, url=url)

    pick_up_tip(rack=tip_rack, well=Tip_Rack_Location[tip_number], location=Pick_Up_Tip_Location,
                pipette=pipette_left, url=url)

    move_to_well(labware=mixing_plate, well=mixing_well, location=Mixing_Location, pipette=pipette_left, url=url)

    # Mixing
    for i in range(2):
        # Aspirate
        aspirate(labware=mixing_plate, well=mixing_well, location=Mixing_Location,
                 flowrate=np.median(flowrate), volume=125, pipette=pipette_left, url=url)
        
        time.sleep(5)

        # Dispense
        dispense(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location,
                 flowrate=np.median(flowrate), volume=130, pipette=pipette_left, url=url)

        # Blowout
        blowout(labware=mixing_plate, well=mixing_well, location=Mixing_Location,
                flowrate=np.median(flowrate), pipette=pipette_left, url=url)

    # Transfer mixture to test plate
    # Aspirate
    aspirate(labware=mixing_plate, well=mixing_well, location=Mixing_Location,
             flowrate=np.median(flowrate), volume=200, pipette=pipette_left, url=url)

    time.sleep(5)

    # Dispense
    dispense(labware=test_plate, well=mixing_well, location=Dispense_Location,
             flowrate=np.median(flowrate), volume=200, pipette=pipette_left, url=url)

    # Blowout
    blowout(labware=test_plate, well=mixing_well, location=Dispense_Location,
            flowrate=np.median(flowrate), pipette=pipette_left, url=url)
    
    move_to_well(labware=test_plate, well=mixing_well, location=Dispense_Leave_Location,
                 speed=5, pipette=pipette_left, url=url)
    
    # Drop tip
    #drop_tip_to_trash(pipette=Pipette_Left_ID, url=commands_url)
    drop_tip(rack=drop_tip_rack, well=Tip_Rack_Location[tip_number], location=Drop_Tip_Location,
                pipette=pipette_left, url=url)
    check_tip()
    time.sleep(900)

"""
Step 1: Create a run
"""
runs_url, run_id = create_run(robot_ip=Robot_IP)

"""
Step 2: Set up commands endpoint
"""
commands_url = f"{runs_url}/{run_id}/commands"

lights_url = f"http://{Robot_IP}:31950/robot/lights"

# Turn on Light
light(status=On, url=lights_url)

"""
Step 3: Load module, labware and pipette
"""
# Load Temperature_Module
#Temperature_Module_ID = load_module(slot=Mixing_Plate_Slot, module=Temperature_Module, url=commands_url)
#print(f"Temperature_Module ID:\n{Temperature_Module_ID}\n")

# Load Labware 1 (Test_Plate)
# Labware_1_ID = load_labware(slot={"moduleId": Temperature_Module_ID}, labware=Test_plate, brand="custom_beta",
#                            url=commands_url)
# print(f"Labware 1 (Test_Plate) ID:\n{Labware_1_ID}\n")

# Load Labware 1 (Test_Plate)
Labware_1_ID = load_labware(slot=Test_Plate_Slot, labware=Test_plate, brand="opentrons",
                            url=commands_url)
print(f"Labware 1 (Test_Plate) ID:\n{Labware_1_ID}\n")

# Load Labware 2 (Mixing_Plate)
Labware_2_ID = load_labware(slot=Mixing_Plate_Slot, labware=Mixing_plate, brand="opentrons", url=commands_url)
print(f"Labware 2 (Mixing_Plate) ID:\n{Labware_2_ID}\n")

# Load Labware 3 (Tip_Rack_1)
Labware_3_ID = load_labware(slot=Tip_Rack_1_Slot, labware=Tip_Rack_1, brand="opentrons", url=commands_url)
print(f"Labware 3 (Tip_Rack_1) ID:\n{Labware_3_ID}\n")
tip_rack = Labware_3_ID

# Load Labware 4 (Tip_Rack_2)
Labware_4_ID = load_labware(slot=Tip_Rack_2_Slot, labware=Tip_Rack_2, brand="opentrons", url=commands_url)
print(f"Labware 4 (Tip_Rack_2) ID:\n{Labware_4_ID}\n")

# Load Labware 5 (Tip_Rack_3)
Labware_5_ID = load_labware(slot=Tip_Rack_3_Slot, labware=Tip_Rack_3, brand="opentrons", url=commands_url)
print(f"Labware 5 (Tip_Rack_3) ID:\n{Labware_5_ID}\n")

# Load Labware 6 (Tip_Rack_4)
Labware_6_ID = load_labware(slot=Tip_Rack_4_Slot, labware=Tip_Rack_4, brand="opentrons", url=commands_url)
print(f"Labware 6 (Tip_Rack_4) ID:\n{Labware_6_ID}\n")
drop_tip_rack = Labware_6_ID

# Load Labware 7 (Tip_Rack_5)
Labware_7_ID = load_labware(slot=Tip_Rack_5_Slot, labware=Tip_Rack_5, brand="opentrons", url=commands_url)
print(f"Labware 7 (Tip_Rack_5) ID:\n{Labware_7_ID}\n")

# Load Labware 8 (Tip_Rack_6)
Labware_8_ID = load_labware(slot=Tip_Rack_6_Slot, labware=Tip_Rack_6, brand="opentrons", url=commands_url)
print(f"Labware 8 (Tip_Rack_6) ID:\n{Labware_8_ID}\n")

# Load Labware 9 (Reservoir)
Labware_9_ID = load_labware(slot=Reservior_Slot, labware=Reservior, brand="opentrons", url=commands_url)
print(f"Labware 9 (Reservoir) ID:\n{Labware_9_ID}\n")

# Load Labware 10 (Fan)
Labware_10_ID = load_labware(slot=Fan_Slot, labware=Fan, brand="opentrons", url=commands_url)
print(f"Labware 10 (Fan) ID:\n{Labware_10_ID}\n")

# Load Labware 11 (Electrolytes)
Labware_11_ID = load_labware(slot=Electrolytes_Slot, labware=Electrolytes, brand="opentrons", url=commands_url)
print(f"Labware 11 (Electrolytes) ID:\n{Labware_11_ID}\n")

# Load Pipette_Right
Pipette_Right_ID = load_pipette(pipette=Pipette_Right, mount="right", url=commands_url)
print(f"Pipette_Right ID:\n{Pipette_Right_ID}\n")

# Load Pipette_Left
Pipette_Left_ID = load_pipette(pipette=Pipette_Left, mount="left", url=commands_url)
print(f"Pipette_Left ID:\n{Pipette_Left_ID}\n")

"""
Step 5: Set target temperature
"""
#set_temperature(id=Temperature_Module_ID, temperature=25, url=commands_url)

"""
Step 6: Home the robot
"""
home_url = f"http://{Robot_IP}:31950/robot/home"
home_robot(url=home_url)

"""
Print all IDs in one place
"""
print("IDs for Commands:")
print(f"Run ID:\n{run_id}\n")
#print(f"Temperature_Module ID:\n{Temperature_Module_ID}\n")
print(f"Labware 1 (Test_Plate) ID:\n{Labware_1_ID}\n")
print(f"Labware 2 (Mixing_Plate) ID:\n{Labware_2_ID}\n")
print(f"Labware 3 (Tip_Rack_1) ID:\n{Labware_3_ID}\n")
print(f"Labware 4 (Tip_Rack_2) ID:\n{Labware_4_ID}\n")
print(f"Labware 5 (Tip_Rack_3) ID:\n{Labware_5_ID}\n")
print(f"Labware 6 (Tip_Rack_4) ID:\n{Labware_6_ID}\n")
print(f"Labware 7 (Tip_Rack_5) ID:\n{Labware_7_ID}\n")
print(f"Labware 8 (Tip_Rack_6) ID:\n{Labware_8_ID}\n")
print(f"Labware 9 (Reservoir) ID:\n{Labware_9_ID}\n")
print(f"Labware 10 (Fan) ID:\n{Labware_10_ID}\n")
print(f"Labware 11 (Electrolytes) ID:\n{Labware_11_ID}\n")
print(f"Pipette_Right ID:\n{Pipette_Right_ID}\n")
print(f"Pipette_Left ID:\n{Pipette_Left_ID}\n")

"""
Step 7: Pick up the electrode and get ready to run
"""
pick_up_tip(rack=Labware_10_ID, well="F10", location=Electrode_Pick_Up_Location, pipette=Pipette_Right_ID, url=commands_url)

print(input("Press Enter if everything is ready to run."))

"""
Step 8: Main workflow
Calibrate cell constant and randomly combine electrolytes and test EIS
"""

#Cell_Constant_Slope, Cell_Constant_Intercept = calibration(pipette_left=Pipette_Left_ID, pipette_right=Pipette_Right_ID,
                                               #           electrolytes=Labware_11_ID, mixing_plate=Labware_2_ID,
                                               #           reservoir=Labware_9_ID, fan=Labware_10_ID, url=commands_url)
                                                          

Cell_Constant_Slope = 1.154831163
Cell_Constant_Intercept = -0.144214763


print(input("Press Enter if everything is ready to run."))

mode =''

for i in range(Samples_Number):
    
    volumes = volume_list[i]
    
    formulation(combination=volumes, pipette_left=Pipette_Left_ID, electrolytes=Labware_11_ID,
                water=12, test_plate=Labware_1_ID, mixing_well=Well_Plate_Location[test_location],
                mixing_plate=Labware_2_ID, flowrate = flowrate, url=commands_url)
        
    potential_oer = 1.8
    potential_her = -0.05
    
    concentrations = []
    for j in range(Electrolytes_Number):
        column_name = 'Liquid_' + str(j + 1)
        Summary[column_name].append(volumes[j])
        concentrations.append(volumes[j] * concentration_list[j])
        Summary[column_name+'_Concentration'].append(concentrations[j])
    Summary['Water'].append(volumes[Electrolytes_Number])
    Summary['Zn_Concentration'].append(sum(concentrations))
    
    Conductivity_List = []
    Potential_List = []
    HER_List = []
    OER_List = []
    
    for j in range(4):

        for k in range(len(concentrations)):
            column_name = 'Liquid_' + str(k + 1)
            Combination[column_name].append(volumes[k])
            Combination[column_name + '_Concentration'].append(concentrations[k])
        Combination['Water'].append(volumes[Electrolytes_Number])
        Combination['Zn_Concentration'].append(sum(concentrations))

        # Move to test plate
        move_to_well(labware=Labware_1_ID, well=Well_Plate_Location[test_location],
                    location=Electrode_Test_Leave_Location, pipette=Pipette_Right_ID, url=commands_url)
        move_to_well(labware=Labware_1_ID, well=Well_Plate_Location[test_location],
                    location=Electrode_Test_Location, pipette=Pipette_Right_ID, url=commands_url)

        # Electrochemical Testing
        impedance(test=test_location, frequency = 1000)
            
        resistance, conductance = data_process_peis(location=Well_Plate_Location[test_location])
        conductivity = to_conductivity(conductance=conductance, slope=Cell_Constant_Slope,
                                        intercept=Cell_Constant_Intercept)
        Combination['Conductivity'].append(conductivity)
        Summary['Conductivity_' + str(j + 1)].append(conductivity)
        Conductivity_List.append(conductivity)

        ocv = data_process_ocv(location=Well_Plate_Location[test_location])

        lsv(test=test_location, ocv = ocv, end_potential=potential_oer)
        
        try:
            onset_potential_OER, end_current_OER = data_process_lsv(location=Well_Plate_Location[test_location], re=resistance,
                                                                    ocv = ocv, mode='OER', end_potential = potential_oer)
        except:
            onset_potential_OER = 0
            end_current_OER = 0
                                                    
        while end_current_OER < 0.0005 and potential_oer < 3.3:
            potential_oer = potential_oer + 0.3

            lsv(test=test_location, ocv = ocv, end_potential=potential_oer)
            
            try:
                onset_potential_OER, end_current_OER = data_process_lsv(location=Well_Plate_Location[test_location], re=resistance,
                                                                        ocv = ocv, mode='OER', end_potential = potential_oer)
            except:
                onset_potential_OER = 0
                end_current_OER = 0

        if end_current_OER < 0.00001:
            onset_potential_OER = 0
        Combination['OER'].append(onset_potential_OER)
        Summary['OER_' + str(j + 1)].append(onset_potential_OER)
        OER_List.append(onset_potential_OER)
            
        move_relative(axis="z", distance=10, speed=10, pipette=Pipette_Right_ID, url=commands_url)
        time.sleep(5)
        move_relative(axis="z", distance=-10, speed=10, pipette=Pipette_Right_ID, url=commands_url)
        
        lsv(test=test_location, ocv = ocv, end_potential=potential_her)

        try: 
            onset_potential_HER, end_current_HER = data_process_lsv(location=Well_Plate_Location[test_location], re=resistance,
                                                                    ocv = ocv, mode='HER', end_potential = potential_her)
        
        except:
            onset_potential_HER = 0
            end_current_HER = 0
            
        while end_current_HER > -0.0005 and potential_her > -0.3:
            potential_her = potential_her - 0.05
                
            lsv(test=test_location, ocv = ocv, end_potential=potential_her)
            
            try:
                onset_potential_HER, end_current_HER = data_process_lsv(location=Well_Plate_Location[test_location], re=resistance,
                                                                        ocv = ocv, mode='HER', end_potential = potential_her)

            except:
                onset_potential_HER = 0
                end_current_HER = 0

        if end_current_HER > -0.00001:
            onset_potential_HER = 0
        Combination['HER'].append(onset_potential_HER)
        Summary['HER_' + str(j + 1)].append(onset_potential_HER)
        HER_List.append(onset_potential_HER)
        
        potential_window = onset_potential_OER - onset_potential_HER
        Combination['Potential'].append(potential_window)
        Summary['Potential_' + str(j + 1)].append(potential_window)
        Potential_List.append(potential_window)

        # Washed by DI_Water and ethanol
        wash(reservoir=Labware_9_ID, pipette=Pipette_Right_ID, mode = 'test', url=commands_url)

        # Dried by Fan
        move_to_well(labware=Labware_10_ID, well='G5', location=Fan_Leave_Location, pipette=Pipette_Right_ID, 
                    url=commands_url)
        move_to_well(labware=Labware_10_ID, well='G5', location=Fan_Location, pipette=Pipette_Right_ID, 
                    url=commands_url)
            
        test_location = test_location + 1
            
        # Save combination to file
        Comb = pd.DataFrame(Combination)
        filename = Path + 'Combination.csv'
        Comb.to_csv(filename, index=False)

        time.sleep(90)
        move_to_well(labware=Labware_10_ID, well='G5', location=Fan_Leave_Location, pipette=Pipette_Right_ID,
                    url=commands_url)
        
    # Calculate average and std
    avg_conductivity = np.mean(Conductivity_List)
    std_conductivity = np.std(Conductivity_List)
    avg_potential = np.mean(Potential_List)
    std_potential = np.std(Potential_List)
    avg_her = np.mean(HER_List)
    std_her = np.std(HER_List)
    avg_oer = np.mean(OER_List)
    std_oer = np.std(OER_List)
    
    # Append to Summary
    Summary["Conductivity_Mean"].append(avg_conductivity)
    Summary['Conductivity_STD'].append(std_conductivity)
    Summary['Potential_Mean'].append(avg_potential)
    Summary['Potential_STD'].append(std_potential)
    Summary['HER_Mean'].append(avg_her)
    Summary['HER_STD'].append(std_her)
    Summary['OER_Mean'].append(avg_oer)
    Summary['OER_STD'].append(std_oer)
    
    # Save Summary to file
    Sum = pd.DataFrame(Summary)
    filename = Path + 'Summary.csv'
    Sum.to_csv(filename, index=False)

    print("Return value: " + str(avg_conductivity) + ", " + str(avg_potential))

# Deactivate Temperature_Module
#deactivate_module(id=Temperature_Module_ID, url=commands_url)

print(input("Press Enter if everything is ready to run."))

"""
Step 9: Home the robot
"""
home_url = f"http://{Robot_IP}:31950/robot/home"
home_robot(url=home_url)

lights_url = f"http://{Robot_IP}:31950/robot/lights"

# Turn off Light
light(status=Off, url=lights_url)

# Stop run
actions_url = f"{runs_url}/{run_id}/actions"
action_payload = json.dumps(
    {"data": {"actionType": "stop"}}
)

r = requests.post(
    url=actions_url,
    headers=Headers,
    data=action_payload
)

print(f"Request status:\n{r}\n{r.text}")

print('run complete!')
