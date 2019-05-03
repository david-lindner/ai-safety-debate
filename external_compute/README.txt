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
Run `ml avail` while accessing a node interactively to see available modules and type `ml NAME` to load NAME.
To use these on runtime, you can put `ml NAME` in a bash scrips (see files in this folder)

In order to get access to sacred, you have to create a virtual environment.
You can do this by starting up the python module you want to use, navigate
to the directory in which you want to run sacred, and run
`python3 -m venv tutorial-env`
to create the environment. Then run
`source bin/activate`
to activate the environment. Install sacred with pip3. Then, get out by running
`deactivate`
In order to make use of this environment later, you can put 
`source bin/activate`
in a bash script (see files in this folder).


