# Read Me

The code in this folder is an example code showing the multi-objective optimization workflow of our SDL. 
Please modify the code according to your own experimental purpose and robotic module, and **review the code 
carefully before you run the code**.


`config`: Configuration `.json` files for different dimensions screening. The final component in the configuration file
is the main solvent in the experiment (in this case, water). If no dilution is required, please remove the final component
in the configuration file and replace the `helper_function.py` line 1454 with

```python
for j in range(len(volumes)):
```



`helper_function.py`: helper function for the SDL, containing OT-2 robot control, EC Biologic potentiostat control, and
optimizer. 

Details for Using HTTP API to control OT-2 robot, see [OT-2 HTTP API](https://github.com/Opentrons/opentrons-integration-tools/tree/main/http-api).

Details of using Python to control EC Biologic potentiostat, see [easy-biologic](https://github.com/bicarlsen/easy-biologic.)

Details of the implementation of Dragonfly for Bayesian optimization, see [Dragonfly](https://github.com/dragonfly/dragonfly).



`main_code.py`: Example main code for the SDL. A five-dimensional mapping was used as the example.
