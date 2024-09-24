"""
NPT MD simulation pipeline

In this python script, we generate a basic atomistic simulation for
a solvated protein structure using the following pipeline:

INPUT: cleaned protein structure 
(Hydrogens + heavy atoms + missing residues added)

- Read in structure
- Solvate structure using tip3p water model
- Paramaterise system using ff14DSB + tip3p forcefield
- Perform energy minimisation
- Slowly ramp up temperature to our target temperature of 300 K
- Perform 1 ns NVT equilibration
- Perform 50 ns NPT MD simulation (implement Monte Carlo barostat)

OUTPUT: .pdb files containing output trajectories + .csv files containing
energy + temperature date

References:
https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/protein_in_water.html
http://www.mdtutorials.com/gmx/lysozyme/index.html

"""
from openmm.app import *
from openmm import *
from openmm.unit import *
import sys
from sys import stdout
import os
import shutil
import time

dt = 2.0000*femtoseconds #Set global 2 fs timestep

def solvate(unsolvated_inpfile, forcefield):
    """
    Function that solvates a clean input protein structure in a cubic box 
    with 15 nm padding, using the Modeller class. 
    Other than specifying the padding, use the default params.

    The input str unsolvated_inpfile should be a .pdb filename
    """

    pdb = PDBFile(unsolvated_inpfile)

    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater() #test for any remaining waters

    modeller.addSolvent(forcefield, padding=1.0*nanometer)
    
    return modeller

def clean_dir():
    """
    Removes old output directories from previous simulations 
    """

    dir_names=['full', 'system_init', 'heating', 'equilibration', 'NPT']
    contents = os.listdir()

    for item in contents:
        for directory in dir_names: 
            if item == directory:
                shutil.rmtree(item)

def heat_system(system): 

    """
    Ramp up temperature slowly from 0 K to 300 K. 
    Increase temperature by 1 K every 1 ps
    """
    print('\nStarting heating...\n')
    os.mkdir('heating')
    simulation.reporters.append(PDBReporter('heating/sys_heating.pdb', 5000))

    simulation.reporters.append(StateDataReporter('heating/energy_heating.csv', 5000,
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
    temperature=True))

    for temp in range(1, 301):

        integrator.setTemperature(temp * kelvin) #set new temperature
        simulation.step(500) #1 ps simulation at given temp

def equilibrate_system(system):

    """
    NVT equilibration at 300 K
    """
    integrator.setTemperature(300.0000 * kelvin) #set to final temperature

    print('\nStarting NVT equilibration at 300 K\n')
    os.mkdir('equilibration')

    simulation.reporters.append(PDBReporter('equilibration/NVT.pdb', 50000))

    simulation.reporters.append(StateDataReporter('equilibration/energy_NVT.csv', 50000,
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
    temperature=True))
    
    simulation.step(250000) #Perform 500 ps equilibration at 300 K

def perform_NPT(system):

    """
    NPT simulation (50 ns)
    """
    print('\nStarting NPT simulation at 300 K, 1 bar\n')
    os.mkdir('NPT')

    simulation.reporters.append(PDBReporter('NPT/NPT.pdb', 50000))

    simulation.reporters.append(StateDataReporter('NPT/energy_NPT.csv', 50000,
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
    temperature=True))

    system.addForce(MonteCarloBarostat(1*bar, 300.0000*kelvin))
    simulation.context.reinitialize(preserveState=True)
    simulation.step(2.5E7) #50 ns 

def main():

    if len(sys.argv) < 2:
        print("Specify cleaned protein structure as input")
        sys.exit(1)  # Exit the script with an error code (1)

    else:
        inp_file = str(sys.argv[1])

    """Define forcefield for protein + water"""
    forcefield =ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')

    """Solvate system in cubic box"""
    pdb_solvated = solvate(inp_file, forcefield)

    """Generate system"""
    system = forcefield.createSystem(pdb_solvated.topology, nonbondedMethod=PME, nonbondedCutoff=1.5*nanometer, constraints=HBonds)    

    """Use Langevin integrator with 1/ps friction coefficient, with initial temp = 0 K"""
    integrator = LangevinMiddleIntegrator(0.0000*kelvin, 1.0000/picosecond, dt)

    """Set up simulation"""
    platform = Platform.getPlatformByName('CUDA') 
    simulation = Simulation(pdb_solvated.topology, system, integrator, platform)
    simulation.context.setPositions(pdb_solvated.positions)
    simulation.minimizeEnergy() #Minimise energy

    """Remove old directories"""
    print('\nRemoving old directories...\n')
    clean_dir()

    """Initialise print reporter for whole simulation"""
    print('\nStarting simulation...\n')
    os.mkdir('full')

    totalSteps = 25425000
    reportInterval = totalSteps/1000 #25000 + 500*300 + 250000 + 2.5E7 timesteps / 1000 (generate 1000 datapoints)

    simulation.reporters.append(PDBReporter('full/system.pdb', reportInterval))

    simulation.reporters.append(StateDataReporter('full/energy.csv', reportInterval, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
        temperature=True, volume=True, density=True, progress=True, elapsedTime=False, totalSteps=totalSteps))

    simulation.reporters.append(StateDataReporter(stdout, reportInterval, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
        temperature=True, volume=True, density=True, progress=True, elapsedTime=False, totalSteps=totalSteps))

    """Report initial system conditions in separate directory"""

    print('\nGenerating initial system...\n')
    os.mkdir('system_init')
    simulation.reporters.append(PDBReporter('system_init/sys_init.pdb', 1000))

    simulation.reporters.append(StateDataReporter('system_init/energy_init.csv', 1000, 
    time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
    temperature=True))
    
    simulation.step(25000) #Perform 50 ps test simulation at 0 K

    heat_system(simulation)

    equilibrate_system(simulation)

    perform_NPT(simulation)

    print("\nSimulation complete!")

start_time = time.time()     

main()

print('Programme run time is:' + "%s seconds" % (time.time() - start_time))
