#!/bin/sh
# Script that uninstall and reinstalls the multiple_clocks repository from the rsa conda environment
# Please use it after syncing the repository with the latest changes

# Check the number of arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <conda_env>"
    exit 1
fi

# Ensure that the rsa conda environment is activated
conda activate $1

# Uninstall the multiple_clocks repository
pip uninstall mc

# Install the multiple_clocks repository
pip install /home/fs0/chx061/multiple_clocks_repo