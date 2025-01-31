# disSolve
by Rhoslyn Coles


A python scripted algorithm which deforms open or closed tubes under an thermodynamic inspired energy.
The algorithm is based upon the technique of simulated annealing; implemented in parallel between replicated systems annealing at the same temperature.

## Installation

The content of the repository needs to be downloaded into a folder. Supposing the standard python (3.9 or higher) packages are installed, the only additional required python package is [mpi4py](https://mpi4py.readthedocs.io/en/stable/).

The stystem energy is computed using the C++ executable **`morph_local`** that is compiled for macos with GCC as compiler. 
If you would like to use this work but wish to replace the program **`morph_local`** with a version compiled on your own machine, you need to download the [AlphaMol software repository](https://github.com/pkoehl/AlphaMol) and follow the installation guidlines. You will need to change the program somewhat to compute with the input files used in this algorithm. Please contact me if you are interested in doing this!

To effectivly intereact with simulation runs (which can take up to a month of computation time) I recommend using **`screen`**, a terminal multiplexer, which allows you to start a session, run your commands, and then detach and reattach to the session later-- which basically means you can check on a simulation conveniently.
If you are using a unix--like system, screen should be already installed, otherwise consult the [screen documentation](https://www.gnu.org/software/screen/).


## Usage

After downloading the repository, to run a simulation experiment you will need to edit the `submitJob.sh` file to set the parameters and then execute the script to begin the experiment.

1. Open the `submitJob.sh` file in a text editor and modify the parameters, see **Editable Parameters** below for an explanation of this.
2. Once edited, make sure the file is executable (if it's not already):
   ```bash
   chmod +x submitJob.sh
   ```
3. In the current directory type
   ```bash
   ./submitJob.sh
   ```

<details>
<summary> <h3><b> Editable Parameters </b> </h3></summary>
<br> <!-- This adds a newline -->


This script is designed to submit a simulation job, create necessary directories, and execute the simulation with specific configurations. Below are the key variables that you can modify in the script.


1. **`structure`**  
   - **Description**: The name of structure you are experimenting with (e.g., `"circle36"`, `"trefoil56"`, etc.). This is a naming variable (job name and directory name) and otherwise has no other function.  
   - **Example**: `structure="circle36"`
  
2. **`inputFile`**
    - **Description**: Path to the text file which describes the initial curve configuration. Since the curve is a polygonal curve, the file lists the three cooridinate values of each vertex of the polygon per line of the file.
    - **Example**: See examples directory 

3. **`overlapRatio`**  
   - **Description**: This value controls the overlap ratio used in the simulation (a small positive float between 0.02 and  0.4). It is used to define the radius of the tubular neiighbourhood surrounding the tube which defines the structure whos geometric measures (volume, surface area etc.) compute the energy.
   - **Example**: `overlapRatio=0.12`

4. **`eta`**  
   - **Description**: A parameter determines the packing fraction of the fluid (a nonzero positive float between 0 amd 0.495) determining the linear coefficients of the four geometric measures which define the energy.  
   - **Example**: `eta=0.35`

5. **`numberParallelProcesses`**  
   - **Description**: The number of parallel processes to use when running the simulation (integer). Typically, you'd adjust this based on the computational resources available.   
   - **Example**: `numberParallelProcesses=10`

6. **`T_0`**  
   - **Description**: The initial temperature for the simulation. Choosing an effective annealing temperature is difficult, for this reason it is possible to run the simulation with the option **`varyT`** (see below). An effective temperature depends upon the parameters **`overlapRatio`** and  **`eta`** as well as the actual shape of the curve and thus is generally dertermine heuristically.
   - **Example**: `T_0=2.0`

7. **`varyT`**  
    - **Description**: A boolean flag (1 or 0) that indicates whether to vary the temperature during the simulation (this is then set such that a deformaiton that increases the energy equal to the median energy increase of the last 5000 energy increasing deformtions is acccepted with probability 0.5, the temperature is adjusted by decreasing this probability. Set this to `1` if you want to vary the temperature, or `0` to keep the temperature constant. If `varyT=1` then the simulation anneals **first** with the temperature set dynamically and then **second** continues to anneal at the user specified temperature.
    - **Example**: `varyT=1`

8. **`T_step`**  
   - **Description**: This controls how the temperature changes over time, the closer the parameter is to one the more gradually the temperature decreases. If the parameter is larger than one the temperature increases during the simulation.
   - **Example**: `T_step=0.95`

8. **`numberSecondsPerTemp`**  
   - **Description**: The number of seconds the simulation computes each systems at a given fixed temperature before exchanging between systems and possibly changing the temperature.
   - **Example**: `numberSecondsPerTemp=2400`

9. **`numberOfRounds`**  
   - **Description**: The number of times the parallel systems will be exchanged or replacated. This means that the total simulation time in sections is  **`numberSecondsPerTemp`** x  **`numberOfRounds`** 
   - **Where to modify**: Modify this value to adjust the number of simulation rounds.  
   - **Example**: `numberOfRounds=35`

10. **`numberSecondsBetweenUpdatingTempByVaryT`**  
    - **Description**: This variable has the same function as the variable **`numberSecondsPerTemp`** used if `varyT=1`.
    - **Example**: `numberSecondsBetweenUpdatingTempByVaryT=1200`

11. **`numberRoundsVaryT`**  
    - **Description**: This variable has the same function as the variable **`numberOfRounds`** used if `varyT=1`.
    - **Example**: `numberRoundsVaryT=25`

</details>


The **`submitJob.sh`** script copies the relevant python code into a subdirectory and initialises the experiment with **`screen`**. Beyond the simulation, the finishing time of the experiment as well as the **`jobName`** are printed to screen. The output of the experiment can be seen by attaching to the job using
```bash
screen -r jobName
```
to leave the simulation terminal window (detach) press `Ctrl + A`, then release both keys and press `D`. 



    
