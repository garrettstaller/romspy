# romspy
An ever-evolving collection of scripts to assist ROMS users with Python workflows. 

## Architecture:
This repo has two main directories, ***exec*** and ***src***. The former holds helpful .py scripts that may be run at command line
either through aliases or as bash. The .py scripts in ***src*** are what the executables are built off and are designed to hold functions 
you may find helpful in the I/O of ROMS-based NetCDFs (those with traditional staggered C-grid ROMS dimensions), plotting and the computation 
of diagnostic variables!

## Getting ROMSPY to Work
Nothing fancy at the moment as this is a very early layout of the functions that may one day make up a greater Python library. There is, however, the possibility
users do not have cmocean - which is imported into src/plot_utils.py. Moreover, to ensure executable scripts may find the /src files, set an environmental 
variable (in your bash,zsh,etc.) called ROMSPY_ROOT - ensure it points to /path/to/**romspy/src**.

You may also find it helpful to turn on vim folding through your vimrc to navigate the scripts easier. 
This may be done by:

`vi ~/.vimrc` from CL
Then in the vimrc:
`set foldmethod=marker`
`set foldmarker=![,!]`
And run `:source` from vim command line to apply
