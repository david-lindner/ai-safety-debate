These are scripts to run the code externally, with slurm, especially on the RCI clusters.
See a tutorial and additional information here: https://login.rci.cvut.cz/wiki/how_to_start

In order to get access to a node interactively, you can run
`srun --pty bash -i`

In order to run complicated things, you can put them in a bash SCRIPT and run them with
`srun OPTIONS /bin/bash SCRIPT &`
You may want to add OPTIONS such as 
--time=HOURS:MINUTES:SECONDS to specify the maximum time you will allocate the node
-o FILENAME, to send stdout to FILENAME
-e FILENAME, to send errors to FILENAME
-p PARTITION, to specify partition to run on. Partitions are gpu, longjobs, cpu (default)
--gres=gpu:N, to allocate N gpus
--cpus-per-task=N, to allocate N cpus

In order to get access to tensorflow, you have to load an appropriate python modules.
Run `ml avail` while accessing a node interactively to see available modules and run `ml NAME` to load NAME.
To use these on runtime, you can put `ml NAME` in a bash script (see files in this folder)

In order to get access to sacred, you need a virtual environment.

To make use of an already existing virtual environment, you only need to run 
`source bin/activate`
while you're in the correct folder. 
To use this during runtime, you can put `source bin/activate` in a bash script (see files in this folder).

If you have to create a new virtual environment, you can do this by: 
	* accessing any node interactively,
	* starting up the model you'll eventually want to use (or another model with the same version of python),
	* navigating to the directory in which you want to use sacred
	* running `python3 -m venv .` to create the environment
	* running `source bin/activate` to activate the environment
	* installing sacred with pip3
		* make sure that you try to import sacred in python to ensure that it's working
		* If there are problems with h5py, try to import h5py directly and install whatever component it's missing.
Once everything is working, you can exit the virtual environment by running
`deactivate`

