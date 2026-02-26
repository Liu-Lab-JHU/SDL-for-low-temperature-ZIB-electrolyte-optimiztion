#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import requests
import json
import easy_biologic as ebl
import easy_biologic.base_programs as blp
import time
import math
import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from argparse import Namespace
import scipy
from dragonfly import load_config_file, multiobjective_maximise_functions
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# IP of OT-2 and Potentiostat
Headers = {"opentrons-version": "3"}

"""
Labware calibrated location.
To prevent collision of the robot, please calibrate the corresponding location 
according to your own robot and labware before run the code.
"""
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
Electrode_Test_Location = {"origin": "top", "offset": {"x": 0, "y": -1.5, "z": 15.5}}
Electrode_Test_Leave_Location = {"origin": "top", "offset": {"x": 0, "y": -1.5, "z": 80}}
Electrode_Calibration_Location = {"origin": "top", "offset": {"x": -0.5, "y": -0.5, "z": -20}}
Electrode_Calibration_Leave_Location = {"origin": "top", "offset": {"x": -0.5, "y": -0.5, "z": 0}}
Wash_Location = {"origin": "top", "offset": {"x": 0, "y": 0, "z": -40}}
Wash_Leave_Location = {"origin": "top", "offset": {"x": 0, "y": 0, "z": 20}}

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
Electrolytes_Location = {1: 'A1', 2: 'A2', 3: 'A3', 4: 'A4', 5: 'A5', 6: 'A6',
                         7: 'A7', 8: 'A8', 9: 'A9', 10: 'A10', 11: 'A11', 12: 'A12'}

class ExperimentContext:
    def __init__(
            self,
            tip_number,
            tip_rack_number,
            labware_ids,
            electrolyte_tip_number=96
    ):
        self.tip_number = tip_number
        self.tip_rack_number = tip_rack_number
        self.labware_ids = labware_ids
        self.electrolyte_tip_number = electrolyte_tip_number
        self.tip_rack = None
        self.drop_tip_rack = None
        self.update_racks()

    def update_racks(self):
        if self.tip_rack_number == 1:
            self.tip_rack = self.labware_ids.get('Tip_Rack_1')
            self.drop_tip_rack = self.labware_ids.get('Tip_Rack_4')
        elif self.tip_rack_number == 2:
            self.tip_rack = self.labware_ids.get('Tip_Rack_2')
            self.drop_tip_rack = self.labware_ids.get('Tip_Rack_5')
        elif self.tip_rack_number == 3:
            self.tip_rack = self.labware_ids.get('Tip_Rack_3')
            self.drop_tip_rack = self.labware_ids.get('Tip_Rack_6')
        elif self.tip_rack_number > 3:
            self.tip_rack = self.labware_ids.get('Tip_Rack_1')
            self.drop_tip_rack = self.labware_ids.get('Tip_Rack_4')
            self.tip_rack_number = 1

    def check_tip(self):
        self.tip_number = self.tip_number - 4
        if self.tip_number < 1:
            self.tip_number = 96
            self.tip_rack_number = self.tip_rack_number + 1
            self.update_racks()
        else:
            self.update_racks()


class Optimizer:
    def __init__(self, config_path, path):
        self.config_path = config_path
        self.path = path
        self.Comb = None
        self.Prior = None
        self.include_columns = None

    def update_data(self, comb, prior, include_columns):
        self.Comb = comb
        self.Prior = prior
        self.include_columns = include_columns

    def objective_function_multi(self, params):
        X = self.Comb[[col for col in self.Comb.columns if col in self.include_columns]].to_numpy()
        y = self.Comb[['Conductivity', 'Potential']].to_numpy()

        kernel = Matern(nu=1.5, length_scale_bounds=(1e-7, 1e7))

        def optimizer(obj_func, initial_theta, bounds):
            # Use scipy.optimize.minimize to find the optimal hyperparameters
            optimization_result = scipy.optimize.minimize(
                obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter': 10000}
            )
            # Return the optimized parameters and the minimum function value
            return optimization_result.x, optimization_result.fun

        # Initialize the GPR model
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer=optimizer)

        # Fit the GP to the data
        gp.fit(X, y)

        params = np.array(params).reshape(1, -1)  # Reshape for sklearn's predict method
        predicted_y = gp.predict(params)
        print(predicted_y[0])

        return predicted_y[0][0], predicted_y[0][1]

    def conductivity_prior_mean(self, params):
        """ Prior mean for the conductivity. """
        X = self.Prior[[col for col in self.Comb.columns if col in self.include_columns]].to_numpy()
        y = self.Prior['Conductivity', 'Potential'].to_numpy()

        kernel = Matern(nu=1.5, length_scale_bounds=(1e-7, 1e7))

        def optimizer(obj_func, initial_theta, bounds):
            # Use scipy.optimize.minimize to find the optimal hyperparameters
            optimization_result = scipy.optimize.minimize(
                obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter': 10000}
            )
            # Return the optimized parameters and the minimum function value
            return optimization_result.x, optimization_result.fun

        # Initialize the GPR model
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer=optimizer)

        # Fit the GP to the data
        gp.fit(X, y)

        params = np.array(params).reshape(1, -1)  # Reshape for sklearn's predict method
        prior_conductivity = gp.predict(params)

        return prior_conductivity[0][0], prior_conductivity[0][1]

    def initialization_multi(self, prior):
        USE_CONDUCTIVITY_PRIOR_MEAN = prior

        SAVE_AND_LOAD_PROGRESS = True

        config = load_config_file(self.config_path)
        opt_method = 'rand'

        options = Namespace(
            report_results_every=1,  # report progress every 1 iterations
        )

        if USE_CONDUCTIVITY_PRIOR_MEAN:
            options.gp_prior_mean = self.conductivity_prior_mean

        if SAVE_AND_LOAD_PROGRESS:
            options.progress_load_from_and_save_to = self.path + 'progress.p'
            options.progress_save_every = 1

        max_capital = 5
        opt_val, opt_pt, history = multiobjective_maximise_functions(
            (self.objective_function_multi, 2), config.domain, max_capital,
            opt_method=opt_method, config=config, options=options

        )

        return opt_val, opt_pt, history

    def optimization_multi(self, method, prior):
        USE_CONDUCTIVITY_PRIOR_MEAN = prior

        SAVE_AND_LOAD_PROGRESS = True

        config = load_config_file(self.config_path)

        opt_method = method

        options = Namespace(
            build_new_model_every=1,  # update the model every 1 iterations
            report_results_every=1,  # report progress every 1 iterations
        )

        if USE_CONDUCTIVITY_PRIOR_MEAN:
            options.gp_prior_mean = self.conductivity_prior_mean

        if SAVE_AND_LOAD_PROGRESS:
            options.progress_load_from_and_save_to = self.path + 'progress.p'
            options.progress_save_every = 1

        max_capital = 1
        opt_val, opt_pt, history = multiobjective_maximise_functions(
            (self.objective_function_multi, 2), config.domain, max_capital,
            opt_method=opt_method, config=config, options=options
        )

        return opt_val, opt_pt, history


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
    print(f"A new run has been created. Run ID:\n{run_id}")
    return runs_url, run_id

def light(status, url):
    """
    Set light status.

    Args:
        status (str): ON or OFF
        url (str): command url
    """
    try:
        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=status)
        print(f"Light was {status}.")
    except:
        print(f"Could not turn light {status}."
              f"Request status:\n{r}\n{r.text}")

def stop_run(run_id, url):
    """
    Stop the run.

    Args:
        run_id (str): the run ID
        url (str): command url
    """
    try:
        actions_url = f"{url}/{run_id}/actions"
        action_payload = json.dumps(
            {"data": {"actionType": "stop"}}
        )

        r = requests.post(
            url=actions_url,
            headers=Headers,
            data=action_payload
        )
        print(f"Run {run_id} has been stopped.")
    except:
        print(f"Could not stop the run. "
              f"Request status:\n{r}\n{r.text}")

def load_module(slot, module, url, verbosity=False):
    """
    Load OT2 module

    Args:
        slot (str): the slot number for placing the module
        module (string): the module load name
        url: command url
        """
    if verbosity:
        print(f"Loading module {module} to slot {slot}...")
    try:
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

        r = requests.post(
            url=url,
            headers=Headers,
            params={"waitUntilComplete": True},
            data=command_payload
        )

        r_dict = json.loads(r.text)
        module_id = r_dict["data"]["result"]["moduleId"]
        print(f"Module {module} has been loaded."
              f"Module ID:\n{module_id}")
        return module_id
    except:
        print(f"Module {module} was failed to load."
              f"Request status:\n{r}\n{r.text}")

def load_labware(slot, labware, brand, url, verbosity=False):
    """
    Load OT2 labware

    Args:
        slot (dict): the slot number for placing the labware
        labware (string): the labware load name
        brand (string): 'opentrons' or 'custom_beta'
        url: command url
    """
    if verbosity:
        print(f"Loading labware {labware} to slot {slot}...")
    try:
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

        r = requests.post(
            url=url,
            headers=Headers,
            params={"waitUntilComplete": True},
            data=command_payload
        )

        r_dict = json.loads(r.text)
        labware_id = r_dict["data"]["result"]["labwareId"]
        print(f"Labware {labware} has been loaded to slot {slot}."
              f"Labware ID:\n{labware_id}")
        return labware_id
    except:
        print(f"Could not load {labware} to slot {slot}."
              f"Request status:\n{r}\n{r.text}")

def load_pipette(pipette, mount, url, verbosity=False):
    """
    Load OT2 pipette

    Args:
        pipette (string): the pipette load name
        mount (string): mounting position, 'left' or 'right'
        url: command url
    """
    if verbosity:
        print(f"Loading pipette {pipette} to {mount}...")
    try:
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

        r = requests.post(
            url=url,
            headers=Headers,
            params={"waitUntilComplete": True},
            data=command_payload
        )

        r_dict = json.loads(r.text)
        pipette_id = r_dict["data"]["result"]["pipetteId"]
        print(f"Pipette {pipette} has been loaded to {mount}."
              f"Pipette ID:\n{pipette_id}")
        return pipette_id
    except:
        print(f"Could not load pipette {pipette} to {mount}."
              f"Request status:\n{r}\n{r.text}")

def set_temperature(id, temperature, url, verbosity=False):
    """
    Set the temperature of the temperature module

    Args:
        id (str): the ID of temperature module
        temperature (float): the target temperature. Unit is ºC
        url (str): command url
    """
    if verbosity:
        print(f"Setting temperature to {temperature} ºC...")
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
        print(f"Temperature was set to {temperature} ºC.")
    except:
        print(f"Could not set temperature to {temperature}."
              f"Response:\n{r}\n{r.text}\n")

def deactivate_temperature_module(id, url):
    """
    deactivate the temperature module

    Args:
        id (str): the ID of the temperature module
        url (str): command url
    """
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
        print(f"Temperature module {id} has been deactivated.")
    except:
        print(f"Could not deactivate temperature module {id}."
              f"Response:\n{r}\n{r.text}\n")

def home_robot(url):
    """
    Home the OT2 robot, including robotic arm and pipettes.

    Args:
        url (str): home url
    """
    print("Homing robot...")
    try:
        command_dict = {"target": "robot"}
        command_payload = json.dumps(command_dict)

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
        print(f"Robot has been homed.")
    except:
        print(f"Could not home robot."
              f"Response:\n{r}\n{r.text}\n")

def pick_up_tip(rack, well, location, pipette, url, verbosity=False):
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
    if verbosity:
        print(f"Picking up tip from {rack} well {well}...")
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
    except:
        print(f"Could not pick up tip from {rack} at {well}."
              f"Response:\n{r}\n{r.text}\n")

def drop_tip(rack, well, location, pipette, url, verbosity=False):
    """
    Drop pipette tip

    Args:
        rack (string): the labware ID of tip rack or trash bin
        well (string): the well to drop the tip
        location (dict): the location to drop the tip
            Default is {"origin": "top", "offset": {"x": 0, "y": 0, "z": 0}}
        pipette (string): the ID of the pipette that is used to drop the tip
    """
    if verbosity:
        print(f"Dropping tip to {rack} well {well}...")
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
    except:
        print(f"Could not drop tip to {rack} {well}."
              f"Response:\n{r}\n{r.text}\n")

def aspirate(labware, well, location, flowrate, volume, pipette, url, verbosity=False):
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
    if verbosity:
        print(f"Aspirating {volume} μL liquid from {labware} well {well}...")
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
    except:
        print(f"Failed to aspirate liquid."
              f"Response:\n{r}\n{r.text}\n")

def touch_tip(labware, well, location, speed, pipette, url, verbosity=False):
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
    if verbosity:
        print(f"Touching tip on {labware} well {well}...")
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
    except:
        print(f"Failed to touch tip."
              f"Response:\n{r}\n{r.text}\n")

def dispense(labware, well, location, flowrate, volume, pipette, url, verbosity=False):
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
    if verbosity:
        print(f"Dispensing {volume} μL liquid to {labware} well {well}...")
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
    except:
        print(f"Failed to dispense liquid."
              f"Response:\n{r}\n{r.text}\n")

def blowout(labware, well, location, flowrate, pipette, url, verbosity=False):
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
    if verbosity:
        print(f"Blowing out liquid from {labware} well {well}...")
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
    except:
        print(f"Failed to blow out liquid."
              f"Response:\n{r}\n{r.text}\n")

def drop_tip_to_trash(pipette, url, verbosity=False):
    """
    Drop the used tip to trash bin.

    Args:
        pipette (str): the ID of the pipette that is used to drop the tip
        url (str): command url
    """
    if verbosity:
        print(f"Dropping tip to trash bin...")
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
    except:
        print(f"Failed to drop tip to trash bin."
              f"Response:\n{r}\n{r.text}\n")


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

def move_to_well(labware, well, location, pipette, url, speed=400, verbosity=False):
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
    if verbosity:
        print(f"Moving pipette to {labware} well {well}...")
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
    except:
        print(f"Could not move pipette to {labware} well {well}."
              f"Response:\n{r}\n{r.text}\n")

def move_relative(axis, distance, pipette, url, speed=400, verbosity=False):
    """
    Move the pipette relative to its current position.

    Args:
        axis (str): direction to move the pipette. 'x', 'y', 'z'
        distance (float): the distance to move the pipette
        pipette (string): the ID of the pipette that is used to move to the well
        url (string): command url
        speed (float): the speed of the pipette movement
    """
    directions = {
        'x': {1: 'right', -1: 'left'},
        'y': {1: 'in', -1: 'out'},
        'z': {1: 'up', -1: 'down'}
    }
    result = directions[axis][1 if distance > 0 else -1]

    if verbosity:
        print(f"Moving pipette {result} ({axis}) for {distance} mm...")
    try:
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

        r = requests.post(
            url=url,
            params={"waitUntilComplete": True},
            headers=Headers,
            data=command_payload
        )
    except:
        print(f"Could not move pipette relatively to {result} ({axis}) for {distance} mm."
              f"Response:\n{r}\n{r.text}\n")

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

    # Washed by Ethanol (Third Column)
    move_to_well(labware=reservoir, well='E8', location=Wash_Leave_Location, pipette=pipette, url=url)
    move_to_well(labware=reservoir, well='E8', location=Wash_Location, pipette=pipette, url=url)
    move_relative(axis="y", distance=10, pipette=pipette, url=url)
    move_relative(axis="y", distance=-20, pipette=pipette, url=url)
    move_relative(axis="y", distance=10, pipette=pipette, url=url)
    time.sleep(5)
    move_to_well(labware=reservoir, well='E8', location=Wash_Leave_Location, pipette=pipette, url=url)

    # Washed by Ethanol (Fourth Column)
    move_to_well(labware=reservoir, well='E11', location=Wash_Leave_Location, pipette=pipette, url=url)
    move_to_well(labware=reservoir, well='E11', location=Wash_Location, pipette=pipette, url=url)
    move_relative(axis="y", distance=10, pipette=pipette, url=url)
    move_relative(axis="y", distance=-20, pipette=pipette, url=url)
    move_relative(axis="y", distance=10, pipette=pipette, url=url)
    time.sleep(5)
    move_to_well(labware=reservoir, well='E11', location=Wash_Leave_Location, pipette=pipette, url=url)


"""
EC Biologic helper function
"""
def impedance(path, elbIp, test, channel=None, vs_OCV=True, voltage=0, init_freq=1000000, final_freq=0.1,
              amplitude=0.1, sweep=False, repeat=10, duration=0, vs_final=False, time_interval=1,
              current_interval=0.001, correction=False, wati=0.1):
    if channel is None:
        channel = [0]
    Biologic=ebl.BiologicDevice(elbIp)

    if vs_OCV:
        # Run OCP test
        save_path = path + Well_Plate_Location[test] + '_OCV.csv'

        params_ocv = {
            'time': 2,
            'time_interval': 1,
            # 'voltage_interval': 0.01
        }

        ocv = blp.OCV(
            Biologic,
            params_ocv,
            channels=channel
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

        voltage=list(voc.values())[0]

    # Run PEIS test
    save_path = path + Well_Plate_Location[test] + '_PEIS.csv'

    params_peis = {
        'voltage': voltage,
        'final_frequency': final_freq,  # frequency unit: Hertz
        'initial_frequency': init_freq,  # frequency unit: Hertz
        'amplitude_voltage': amplitude,  # voltage unit: Volt
        'frequency_number': np.log10(init_freq/final_freq)*20,
        'duration': duration,  # time unit: second
        'vs_final': vs_final,
        'time_interval': time_interval,  #time unit: second
        'current_interval': current_interval,
        # 'sweep': 'log',
        'repeat': repeat,
        'correction': correction,
        'wait': wati
    }

    peis = blp.PEIS(
        Biologic,
        params_peis,
        channels=channel
    )

    print("Running PEIS for " + Well_Plate_Location[test] + "...")

    peis.run('data')
    peis.save_data(save_path)

def lsv(path, elbIp, test, channel=None, vs_OCV=True, E_start=0, E_end=1, rate=0.01,
        step=0.001, average=False, I_begin=0.5, I_end=1.0):
    if channel is None:
        channel = [0]
    Biologic=ebl.BiologicDevice(elbIp)

    if vs_OCV:
        # Run OCP test
        save_path = path + Well_Plate_Location[test] + '_OCV.csv'

        params_ocv = {
            'time': 2,
            'time_interval': 1,
            # 'voltage_interval': 0.01
        }

        ocv = blp.OCV(
            Biologic,
            params_ocv,
            channels=channel
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

        E_start=list(voc.values())[0]

    # Run LSV test
    if E_end > 0:
        save_path = path + Well_Plate_Location[test] + '_LSV_OER.csv'
        if E_end > 2.7:
            E_start = E_end - 1.5
    elif E_end < 0:
        E_start = E_end + 1.2
        save_path = path + Well_Plate_Location[test] + '_LSV_HER.csv'

    params_lsv = {
        'start': E_start,
        'end': E_end,
        'E2': E_end,
        'Ef': E_end,
        # 'vs_initial': [True, False, False, True, True],
        'rate': rate,  # unit: V/s. We can change the unit on base_programs if we want.
        'step': step,  # step=dEN/1000
        'N_Cycles': 0,
        'average_over_dE': average,
        'begin_measuring_I': I_begin,
        'End_measuring_I': I_end
    }

    lsv = blp.CV(
        Biologic,
        params_lsv,
        channels=channel
    )

    print("Running LSV for " + Well_Plate_Location[test] + " to " + str(E_end) + " V")

    lsv.run('data')
    lsv.save_data(save_path)

"""
Data processing helper function
"""
def data_process_peis(path, test):
    print("Processing PEIS for " + test)

    peis_filename = path + test + '_PEIS.csv'
    df_peis = pd.read_csv(peis_filename, sep=',')
    df_peis['|Z| [Ohm]'] = df_peis['Impedance modulus']
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
    peis_filename = path + test + '_PEIS_Calculated.csv'
    peis_data_output.to_csv(peis_filename, index=False)

    if re != 0:
        plt.plot(df_peis['ReZ [Ohm]'], df_peis['-ImZ [Ohm]'], 'ko',
                 df_peis_range['ReZ [Ohm]'], ((df_peis_range['ReZ [Ohm]'] * peis_slope) + peis_intercept), '--r')
        plt.xlabel('ReZ [Ohm]')
        plt.ylabel('-ImZ [Ohm]')
        plt.title(test + '_PEIS')
        plt.savefig(path + test + '_PEIS.svg')
        plt.close()
    else:
        plt.plot(df_peis['ReZ [Ohm]'], df_peis['-ImZ [Ohm]'], 'ko')
        plt.xlabel('ReZ [Ohm]')
        plt.ylabel('-ImZ [Ohm]')
        plt.title(test + '_PEIS')
        plt.savefig(path + test + '_PEIS.svg')
        plt.close()

    return re, con

def data_process_lsv(path, location, mode, E_end, re=0, ir_correction=True, correction_range=0):
    print("Processing LSV for " + location)

    # Determine resistance
    if mode == 'OER':
        lsv_filename = path + location + '_LSV_OER.csv'
    elif mode == 'HER':
        lsv_filename = path + location + '_LSV_HER.csv'

    df = pd.read_csv(lsv_filename, sep=',')
    df.drop(df.index[:60], inplace=True)

    try:
        # iR correction
        df['Voltage_Corrected [V]'] = df['Voltage'] - df['Current'] * re * 0.1
        df['Current'] = savgol_filter(df['Current'], window_length=100, polyorder=2)

        potential = pd.DataFrame()

        if E_end > 0:
            if E_end > 2.7:
                start_potential = E_end - 1.5
        elif E_end < 0:
            start_potential = E_end + 1.2
        
        # Find derivative point
        if mode == 'HER':
            potential = (df[(df['Voltage_Corrected [V]'] < start_potential) & (
                    df['Voltage_Corrected [V]'] > E_end)].reset_index())
            threshold = 2

        elif mode == 'OER':
            potential = (df[(df['Voltage_Corrected [V]'] < E_end) & (
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
            plt.savefig(path + location + '_LSV_OER.svg')
            plt.close()
        elif mode == 'HER':
            plt.savefig(path + location + '_LSV_HER.svg')
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
        lsv_filename_corrected = path + location + '_LSV_OER_Corrected.csv'
    elif mode == 'HER':
        lsv_filename_corrected = path + location + '_LSV_HER_Corrected.csv'
        
    data_output.to_csv(lsv_filename_corrected, index=False)
    
    try:
        end_current = df['Current'].iloc[-1]
    except:
        end_current = 0

    return onset_potential, end_current

def data_process_ocv(path, location):
    print("Processing OCV for " + location)

    ocv_filename = path + location + '_OCV.csv'
    df_ocv = pd.read_csv(ocv_filename, sep=',')
    ocv = df_ocv['Voltage [V]'].mean()

    return ocv

def to_conductivity(conductance, slope, intercept):
    if conductance <= 0:
        conductivity = 0
    else:
        conductivity = 10**((np.log10(conductance) * slope) + intercept)
    return conductivity


def calibration(path, elbIp, electrode, mixing_plate, reservoir, fan, url, room_temperature=20, start_location=1):
    # Calibration curve
    Calibration = {'Temperature': [room_temperature], 'Conductivity_1': [], 'Conductance_1': [],
                   'Conductivity_2': [], 'Conductance_2': [], 'Conductivity_3': [], 'Conductance_3': [],
                   'Slope': [], 'Intercept': []}

    standard_conductances = []
    
    calibration_location = [start_location, start_location+1, start_location+2, start_location+3]
    for i in calibration_location:
        # Move to test plate
        move_to_well(labware=mixing_plate, well=Well_Plate_Location[i], location=Electrode_Calibration_Leave_Location,
                     pipette=electrode, url=url)
        move_to_well(labware=mixing_plate, well=Well_Plate_Location[i], location=Electrode_Calibration_Location,
                     pipette=electrode, url=url)

        # Electrochemical Testing and electrode regeneration
        impedance(path=path, elbIp=elbIp, test=i, final_freq = 100)
        move_to_well(labware=mixing_plate, well=Well_Plate_Location[i], location=Electrode_Calibration_Leave_Location,
                     pipette=electrode, url=url)
        # Washed by DI_Water and ethanol
        wash(reservoir=reservoir, pipette=electrode, mode='calibration', url=url)
        # Dried by Fan
        move_to_well(labware=fan, well='G5', location=Fan_Leave_Location, pipette=electrode, url=url)
        move_to_well(labware=fan, well='G5', location=Fan_Location, pipette=electrode, url=url)

        if i > start_location:
            _, standard_conductance = data_process_peis(path,test=Well_Plate_Location[i])
            standard_conductances.append(standard_conductance)

            number = i-start_location
            Calibration['Conductance_'+str(number)] = standard_conductance
        
        time.sleep(90)
        move_to_well(labware=fan, well='G5', location=Fan_Leave_Location, pipette=electrode, url=url)

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
    filename = path + 'Calibration.csv'
    Cali.to_csv(filename, index=False)

    return log_slope, log_intercept

def formulation(combination, pipette, electrolyte_tiprack, electrolyte_number,
                electrolytes, water, test_plate, mixing_well, mixing_plate,
                flowrates, url, context, electrolyte_start_column=6):
    electrolyte_tip_rack = None
    if context.tip_rack_number == 1:
        electrolyte_tip_rack = electrolyte_tiprack[0]
    elif context.tip_rack_number == 2:
        electrolyte_tip_rack = electrolyte_tiprack[1]
    elif context.tip_rack_number == 3:
        electrolyte_tip_rack = electrolyte_tiprack[2]

    # Pick up tip
    pick_up_tip(rack=electrolyte_tip_rack, well=Tip_Rack_Location[context.electrolyte_tip_number],
                location=Pick_Up_Tip_Location, pipette=pipette, url=url)
    # Aspirate Water
    aspirate(labware=electrolytes, well=Electrolytes_Location[water], location=Electrolyte_Aspirate_Location,
             flowrate=50, volume=combination[electrolyte_number], pipette=pipette, url=url)
    time.sleep(1)
    # Dispense Water
    dispense(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location,
             flowrate=50, volume=combination[electrolyte_number], pipette=pipette, url=url)
    # Blowout Water
    blowout(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location,
            flowrate=50, pipette=pipette, url=url)
    # Touch Tip
    touch_tip(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location, speed=10,
              pipette=pipette, url=url)
    drop_tip(rack=electrolyte_tip_rack, well=Tip_Rack_Location[context.electrolyte_tip_number],
            location=Drop_Tip_Location, pipette=pipette, url=url)
    time.sleep(1)

    # Add Electrolytes
    for i in range(electrolyte_number):
        if combination[i] > 0 and i < 2:
            # Pick up tip
            pick_up_tip(rack=electrolyte_tip_rack, well=Tip_Rack_Location[(context.electrolyte_tip_number - (8 * (i + 1)))],
                     location=Pick_Up_Tip_Location, pipette=pipette, url=url)
            # Aspirate
            aspirate(labware=electrolytes, well=Electrolytes_Location[i + electrolyte_start_column],
                     location=Electrolyte_Leave_Location, flowrate=150, volume=5, pipette=pipette, url=url)
            move_to_well(labware=electrolytes, well=Electrolytes_Location[i + electrolyte_start_column],
                         location=Electrolyte_Aspirate_Location, speed=10, pipette=pipette, url=url)
            aspirate(labware=electrolytes, well=Electrolytes_Location[i + electrolyte_start_column],
                     location=Electrolyte_Aspirate_Location, flowrate=flowrates[i], volume=combination[i],
                     pipette=pipette, url=url)
            time.sleep(5)
            move_to_well(labware=electrolytes, well=Electrolytes_Location[i + electrolyte_start_column],
                         location=Electrolyte_Leave_Location, speed=1, pipette=pipette, url=url)
            time.sleep(5)
            # Dispense
            dispense(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location,
                     flowrate=flowrates[i], volume=combination[i]+5, pipette=pipette, url=url)
            # Blowout
            blowout(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location,
                    flowrate=flowrates[i], pipette=pipette, url=url)
            # Touch Tip
            touch_tip(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location, speed=10,
                      pipette=pipette, url=url)
            time.sleep(5)
            drop_tip(rack=electrolyte_tip_rack, well=Tip_Rack_Location[(context.electrolyte_tip_number - (8 * (i + 1)))],
                    location=Drop_Tip_Location, pipette=pipette, url=url)

    pick_up_tip(rack=electrolyte_tip_rack, well=Tip_Rack_Location[context.tip_number], location=Pick_Up_Tip_Location,
                pipette=pipette, url=url)
    move_to_well(labware=mixing_plate, well=mixing_well, location=Mixing_Location, pipette=pipette, url=url)
    # Mixing
    for i in range(2):
        # Aspirate
        aspirate(labware=mixing_plate, well=mixing_well, location=Mixing_Location,
                 flowrate=np.median(flowrates), volume=125, pipette=pipette, url=url)
        time.sleep(5)
        # Dispense
        dispense(labware=mixing_plate, well=mixing_well, location=Mixing_Leave_Location,
                 flowrate=np.median(flowrates), volume=130, pipette=pipette, url=url)
        # Blowout
        blowout(labware=mixing_plate, well=mixing_well, location=Mixing_Location,
                flowrate=np.median(flowrates), pipette=pipette, url=url)

    # Transfer mixture to test plate
    # Aspirate
    aspirate(labware=mixing_plate, well=mixing_well, location=Mixing_Location,
             flowrate=np.median(flowrates), volume=200, pipette=pipette, url=url)
    time.sleep(5)
    # Dispense
    dispense(labware=test_plate, well=mixing_well, location=Dispense_Location,
             flowrate=np.median(flowrates), volume=200, pipette=pipette, url=url)
    # Blowout
    blowout(labware=test_plate, well=mixing_well, location=Dispense_Location,
            flowrate=np.median(flowrates), pipette=pipette, url=url)
    move_to_well(labware=test_plate, well=mixing_well, location=Dispense_Leave_Location,
                 speed=5, pipette=pipette, url=url)
    # Drop tip
    drop_tip_to_trash(pipette=pipette, url=url)
    context.check_tip()


def process_sample(volumes, mode, context, test_location, concentration_list,
                   electrolytes_number, Summary, Combination, pipette_left,
                   pipette_right, tip_racks, electrolytes_labware, test_plate,
                   mixing_plate, reservoir, fan, flowrates, url, path, elb_ip,
                   cell_constant_slope, cell_constant_intercept, rest=900):
    
    concentrations = []
    for j in range(len(volumes)-1):
        column_name = 'Liquid_' + str(j + 1)
        volumes[j] = int((volumes[j] / concentration_list[j]) * 250)
        if volumes[j] < 5:
            volumes[j] = 0
        Summary[column_name].append(volumes[j])
    volumes.append(250-sum(volumes))
    Summary['Water'].append(volumes[electrolytes_number])

    for j in range(electrolytes_number):
        column_name = 'Liquid_' + str(j + 1)
        concentrations.append(volumes[j] / 250 * concentration_list[j])
        Summary[column_name+'_Concentration'].append(concentrations[j])
    Summary['Zn_Concentration'].append(sum(concentrations))
    Summary['Mode'].append(mode)

    formulation(
        combination=volumes, pipette=pipette_left,
        electrolyte_tiprack=tip_racks,
        electrolyte_number=electrolytes_number, electrolytes=electrolytes_labware,
        water=12, test_plate=test_plate, mixing_well=Well_Plate_Location[test_location],
        mixing_plate=mixing_plate, flowrates=flowrates, url=url, context=context
    )
    
    potential_oer = 1.8
    potential_her = -0.05

    Conductivity_List = []
    Potential_List = []
    HER_List = []
    OER_List = []

    for j in range(4):
        for k in range(len(concentrations)):
            column_name = 'Liquid_' + str(k + 1)
            Combination[column_name].append(volumes[k])
            Combination[column_name + '_Concentration'].append(concentrations[k])
        Combination['Water'].append(volumes[electrolytes_number])
        Combination['Zn_Concentration'].append(sum(concentrations))
        Combination['Mode'].append(mode)

        # Rest before testing
        time.sleep(rest)

        # Move to test plate
        move_to_well(labware=test_plate, well=Well_Plate_Location[test_location],
                     location=Electrode_Test_Leave_Location, pipette=pipette_right, url=url)
        move_to_well(labware=test_plate, well=Well_Plate_Location[test_location],
                     location=Electrode_Test_Location, pipette=pipette_right, url=url)

        # Electrochemical Testing
        impedance(path=path, elbIp=elb_ip, test=test_location, init_freq=1000)
        
        resistance, conductance = data_process_peis(path, test=Well_Plate_Location[test_location])
        conductivity = to_conductivity(conductance=conductance, slope=cell_constant_slope,
                                        intercept=cell_constant_intercept)
        Combination['Conductivity'].append(conductivity)
        Summary['Conductivity_' + str(j + 1)].append(conductivity)
        Conductivity_List.append(conductivity)

        lsv(path=path, elbIp=elb_ip, test=test_location, E_end=potential_oer)
        
        onset_potential_OER, end_current_OER = data_process_lsv(path=path, location=Well_Plate_Location[test_location], re=resistance,
                                                mode='OER', E_end = potential_oer)
                                                
        while end_current_OER < 0.0005 and potential_oer < 3.3:
            potential_oer = potential_oer + 0.3
            
            lsv(path=path, elbIp=elb_ip, test=test_location, E_end=potential_oer)
        
            onset_potential_OER, end_current_OER = data_process_lsv(path=path, location=Well_Plate_Location[test_location], re=resistance,
                                                                    mode='OER', E_end = potential_oer)

        if end_current_OER < 0.00001:
            onset_potential_OER = 0
        Combination['OER'].append(onset_potential_OER)
        Summary['OER_' + str(j + 1)].append(onset_potential_OER)
        OER_List.append(onset_potential_OER)
        
        move_relative(axis="z", distance=10, speed=10, pipette=pipette_right, url=url)
        time.sleep(5)
        move_relative(axis="z", distance=-10, speed=10, pipette=pipette_right, url=url)

        lsv(path=path, elbIp=elb_ip, test=test_location, E_end=potential_her)
        
        onset_potential_HER, end_current_HER = data_process_lsv(path=path, location=Well_Plate_Location[test_location], re=resistance,
                                                                mode='HER', E_end = potential_her)
        
        while end_current_HER > -0.0005 and potential_her > -0.3:
            potential_her = potential_her - 0.05
            
            lsv(path=path, elbIp=elb_ip, test=test_location, E_end=potential_her)
        
            onset_potential_HER, end_current_HER = data_process_lsv(path=path, location=Well_Plate_Location[test_location], re=resistance,
                                                                    mode='HER', E_end = potential_her)

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
        wash(reservoir=reservoir, pipette=pipette_right, mode = 'test', url=url)

        # Dried by Fan
        move_to_well(labware=fan, well='G5', location=Fan_Leave_Location, pipette=pipette_right, 
                     url=url)
        move_to_well(labware=fan, well='G5', location=Fan_Location, pipette=pipette_right, 
                     url=url)
        
        test_location = test_location + 1
        
        # Save combination to file
        Comb_df = pd.DataFrame(Combination)
        filename = path + 'Combination.csv'
        Comb_df.to_csv(filename, index=False)

        time.sleep(90)
        move_to_well(labware=fan, well='G5', location=Fan_Leave_Location, pipette=pipette_right,
                    url=url)
    
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
    Summary['Conductivity_Mean'].append(avg_conductivity)
    Summary['Conductivity_STD'].append(std_conductivity)
    Summary['Potential_Mean'].append(avg_potential)
    Summary['Potential_STD'].append(std_potential)
    Summary['HER_Mean'].append(avg_her)
    Summary['HER_STD'].append(std_her)
    Summary['OER_Mean'].append(avg_oer)
    Summary['OER_STD'].append(std_oer)

    # Save Summary to file
    Sum_df = pd.DataFrame(Summary)
    filename = path + 'Summary.csv'
    Sum_df.to_csv(filename, index=False)
    
    return test_location
