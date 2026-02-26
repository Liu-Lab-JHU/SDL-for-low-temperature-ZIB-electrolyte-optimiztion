from __future__ import print_function

import requests
import json
import os
import pandas as pd
import helper_function as hf

# IP of OT-2 and Potentiostat
Robot_IP = "RobotIP"    #OT-2 IP
elbIp="ECLab_IP"    #EC Biologic IP

On = json.dumps({"on": True, "waitUntilComplete": True})
Off = json.dumps({"on": False, "waitUntilComplete": True})

# Data Saving Directory
path = "data/path/"
# Configuration data
configuration = "path/config/config.json"

# Labware Variables
Tip_Rack_1_Name = "opentrons_96_tiprack_300ul"
Tip_Rack_2_Name = "opentrons_96_tiprack_300ul"
Tip_Rack_3_Name = "opentrons_96_tiprack_300ul"
Tip_Rack_4_Name = "opentrons_96_tiprack_300ul"
Tip_Rack_5_Name = "opentrons_96_tiprack_300ul"
Tip_Rack_6_Name = "opentrons_96_tiprack_300ul"
Test_plate = "nest_96_wellplate_200ul_flat"
Mixing_plate = "nest_96_wellplate_200ul_flat"
Reservoir = 'nest_96_wellplate_2ml_deep'
Fan = "opentrons_96_tiprack_300ul"
Electrolytes = 'nest_96_wellplate_2ml_deep'
Pipette_Left = "p300_multi_gen2"
Pipette_Right = "p300_single_gen2"
# Pipette_Right = "p300_single"

# Labware Location
Test_Plate_Slot = {"slotName": "7"}
Mixing_Plate_Slot = {"slotName": "2"}
Tip_Rack_1_Slot = {"slotName": "6"}
Tip_Rack_2_Slot = {"slotName": "9"}
Tip_Rack_3_Slot = {"slotName": "11"}
Tip_Rack_4_Slot = {"slotName": "4"}
Tip_Rack_5_Slot = {"slotName": "5"}
Tip_Rack_6_Slot = {"slotName": "8"}
Reservoir_Slot = {"slotName": "1"}
Fan_Slot = {"slotName": "10"}
Electrolytes_Slot = {"slotName": "3"}

# Samples Variable
room_temperature = 21.2
Samples_Number = 4                                  # maximum = 23
start_location = 1
Electrolytes_Number = 5
electrolyte_tip_number = 96
tip_number = electrolyte_tip_number - 8 * (Electrolytes_Number + 1)
tip_rack_number = 1
electrolyte_start_column = 1
flowrates = [5, 5, 10, 10, 50]                      #flow rate of formulation. Replace it if needed.
concentration_list = [15, 10, 3.458, 1.913, 0.219]  #concentration of each component stock solution.
                                                    #replace the concentration to your own example.

# Test Plate Location
Well_Plate_Location = hf.Well_Plate_Location

# Tip Rack Location
Tip_Rack_Location = hf.Tip_Rack_Location

if not os.path.exists(path + 'Combination.csv'):
    Combination = {'Mode': [0], 'Conductivity': [0], 'Potential': [0], 'HER': [0], 'OER': [0]}
    for i in range(Electrolytes_Number):
        Combination['Liquid_' + str(i + 1)] = [0]
        Combination['Liquid_' + str(i + 1) + '_Concentration'] = [0]
    Combination['Water'] = [0]
    Combination['Zn_Concentration'] = [0]
    Comb = pd.DataFrame(Combination)
else:
    Comb = pd.read_csv(path + 'Combination.csv')
    Combination = Comb.to_dict(orient='list')

if os.path.exists(path + 'Prior.csv'):
    Prior = pd.read_csv(path + 'Prior.csv')
    prior = True
    print('Prior Exists')
else:
    Prior = None
    prior = False

if not os.path.exists(path + 'Summary.csv'):
    Summary = {'Mode':[], 'Conductivity_Mean': [], 'Conductivity_STD': [], 'Potential_Mean': [], 'Potential_STD': [],
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
else:
    Sum = pd.read_csv(path + 'Summary.csv')
    Summary = Sum.to_dict(orient='list')

include_columns = []
for i in range(Electrolytes_Number):
    include_columns.append('Liquid_' + str(i+1) + '_Concentration')

# Initialize Optimizer
optimizer = hf.Optimizer(configuration, path)
optimizer.update_data(Comb, Prior, include_columns)

"""
Step 1: Create a run
"""
runs_url, run_id = hf.create_run(robot_ip=Robot_IP)

"""
Step 2: Set up commands endpoint
"""
commands_url = f"{runs_url}/{run_id}/commands"

lights_url = f"http://{Robot_IP}:31950/robot/lights"

# Turn on Light
hf.light(status=On, url=lights_url)

"""
Step 3: Load module, labware and pipette
"""
# Load Labware 1 (Test_Plate)
Labware_1_ID = hf.load_labware(slot=Test_Plate_Slot, labware=Test_plate, brand="opentrons", url=commands_url)

# Load Labware 2 (Mixing_Plate)
Labware_2_ID = hf.load_labware(slot=Mixing_Plate_Slot, labware=Mixing_plate, brand="opentrons", url=commands_url)

# Load Labware 3 (Tip_Rack_1)
Tip_Rack_1_ID = hf.load_labware(slot=Tip_Rack_1_Slot, labware=Tip_Rack_1_Name, brand="opentrons", url=commands_url)

# Load Labware 4 (Tip_Rack_2)
Tip_Rack_2_ID = hf.load_labware(slot=Tip_Rack_2_Slot, labware=Tip_Rack_2_Name, brand="opentrons", url=commands_url)

# Load Labware 5 (Tip_Rack_3)
Tip_Rack_3_ID = hf.load_labware(slot=Tip_Rack_3_Slot, labware=Tip_Rack_3_Name, brand="opentrons", url=commands_url)

# Load Labware 6 (Tip_Rack_4)
Tip_Rack_4_ID = hf.load_labware(slot=Tip_Rack_4_Slot, labware=Tip_Rack_4_Name, brand="opentrons", url=commands_url)

# Load Labware 7 (Tip_Rack_5)
Tip_Rack_5_ID = hf.load_labware(slot=Tip_Rack_5_Slot, labware=Tip_Rack_5_Name, brand="opentrons", url=commands_url)

# Load Labware 8 (Tip_Rack_6)
Tip_Rack_6_ID = hf.load_labware(slot=Tip_Rack_6_Slot, labware=Tip_Rack_6_Name, brand="opentrons", url=commands_url)

# Load Labware 9 (Reservoir)
Labware_9_ID = hf.load_labware(slot=Reservoir_Slot, labware=Reservoir, brand="opentrons", url=commands_url)

# Load Labware 10 (Fan)
Labware_10_ID = hf.load_labware(slot=Fan_Slot, labware=Fan, brand="opentrons", url=commands_url)

# Load Labware 11 (Electrolytes)
Labware_11_ID = hf.load_labware(slot=Electrolytes_Slot, labware=Electrolytes, brand="opentrons", url=commands_url)

# Load Pipette_Right
Pipette_Right_ID = hf.load_pipette(pipette=Pipette_Right, mount="right", url=commands_url)

# Load Pipette_Left
Pipette_Left_ID = hf.load_pipette(pipette=Pipette_Left, mount="left", url=commands_url)

"""
Step 4: Home the robot
"""
home_url = f"http://{Robot_IP}:31950/robot/home"
hf.home_robot(url=home_url)

"""
Step 5: Pick up the electrode and get ready to run
"""
hf.pick_up_tip(rack=Labware_10_ID, well="F10", location=hf.Electrode_Pick_Up_Location,
               pipette=Pipette_Right_ID, url=commands_url)

print(input("Initialization complete. Press Enter if everything is ready to run."))

"""
Step 6: Main workflow
"""
check_calibration=input("Is cell constant calibration required (Y/N)? "
                        "Press N to input cell constant. Press any key else to run cell constant calibration.")
if check_calibration == "N":
    test_location = start_location
    #input cell constant slope and intercept
    Cell_Constant_Slope = float(input("Please input the slope of the cell constant calibration curve."))
    Cell_Constant_Intercept = float(input("Please input the intercept of the cell constant calibration curve."))
else:
    test_location = start_location + 4
    #Cell constant calibration.
    Cell_Constant_Slope, Cell_Constant_Intercept = hf.calibration(path=path, elbIp=elbIp, electrode=Pipette_Right_ID,
                                                                  mixing_plate=Labware_2_ID, reservoir=Labware_9_ID,
                                                                  fan=Labware_10_ID, url=commands_url)

mode =''

# Initialize Experiment Context
labware_ids = {
    'Tip_Rack_1': Tip_Rack_1_ID,
    'Tip_Rack_2': Tip_Rack_2_ID,
    'Tip_Rack_3': Tip_Rack_3_ID,
    'Tip_Rack_4': Tip_Rack_4_ID,
    'Tip_Rack_5': Tip_Rack_5_ID,
    'Tip_Rack_6': Tip_Rack_6_ID
}
context = hf.ExperimentContext(tip_number, tip_rack_number, labware_ids)

tip_racks = [Tip_Rack_1_ID, Tip_Rack_2_ID, Tip_Rack_3_ID]

if os.path.exists(path + 'progress.p'):
    print("Progress file found. Skipping initialization.")
    for i in range(Samples_Number):
        mode = 'bo'
        opt_val, opt_pt, history = optimizer.optimization_multi(mode, prior)
        volumes = history.query_points[1][0]
        test_location = hf.process_sample(
            volumes, mode, context, test_location,
            concentration_list, Electrolytes_Number,
            Summary, Combination,
            Pipette_Left_ID, Pipette_Right_ID,
            tip_racks,
            Labware_11_ID,
            Labware_1_ID, Labware_2_ID, Labware_9_ID, Labware_10_ID,
            flowrates, commands_url, path, elbIp,
            Cell_Constant_Slope, Cell_Constant_Intercept
        )

else:
    print("No progress file found. Initializing.")
    mode = 'rand'
    opt_val, opt_pt, history = optimizer.initialization_multi(prior)
    for i in range(5):
        volumes = history.query_points[i][0]
        test_location = hf.process_sample(
            volumes, mode, context, test_location,
            concentration_list, Electrolytes_Number,
            Summary, Combination,
            Pipette_Left_ID, Pipette_Right_ID,
            tip_racks,
            Labware_11_ID,
            Labware_1_ID, Labware_2_ID, Labware_9_ID, Labware_10_ID,
            flowrates, commands_url, path, elbIp,
            Cell_Constant_Slope, Cell_Constant_Intercept
        )

    Samples_Number = Samples_Number - 5
    for i in range(Samples_Number):
        mode = 'bo'
        opt_val, opt_pt, history = optimizer.optimization_multi(mode, prior)
        volumes = history.query_points[1][0]
        test_location = hf.process_sample(
            volumes, mode, context, test_location,
            concentration_list, Electrolytes_Number,
            Summary, Combination,
            Pipette_Left_ID, Pipette_Right_ID,
            tip_racks,
            Labware_11_ID,
            Labware_1_ID, Labware_2_ID, Labware_9_ID, Labware_10_ID,
            flowrates, commands_url, path, elbIp,
            Cell_Constant_Slope, Cell_Constant_Intercept
        )

"""
Step 7: SDL finalization
"""
home_url = f"http://{Robot_IP}:31950/robot/home"
hf.home_robot(url=home_url)

# Turn off Light
hf.light(status=Off, url=lights_url)

# Stop run
hf.stop_run(run_id=run_id, url=runs_url)

print('Run complete!')
