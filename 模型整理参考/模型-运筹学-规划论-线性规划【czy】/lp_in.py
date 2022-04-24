'''
The problem is read from:
    /Applications/copt30/examples/python/lp_ex1.mps
'''

# Import the Python interface library
from coptpy import *

# Create COPT environment
env = Envr()

# Create COPT model
model = env.createModel("lp_ex1")

# Read MPS model
model.read("lp_ex1.mps")

# Set log file
m.setLogFile("lp_ex1.log")

# Set parameter
m.setParam(COPT.Param.TimeLimit, 10.0)

# Set solving methos to Barrier
m.setParam(COPT.Param.LpMethod, 2)

# Solve the model
model.solve()

# Analyze solution
if model.status == COPT.OPTIMAL:
    # Get objective value
    print("Objective value: {}".format(model.objval))
    allvars = model.getVars()

    # Get valiable solution
    print("Variable solution:")
    for var in allvars:
        print(" x[{0}]: {1}".format(var.index, var.x))

    # Get variable basis status
    print("Variable basis status:")
    for var in allvars:
        print(" x[{0}]: {1}".format(var.index, var.basis))

# Write MPS model, solution, basis and parameters files
model.write("lp_ex1.mps")
model.write("lp_ex1.sol")
model.write("lp_ex1.bas")
model.write("lp_ex1.par")
