set -e #End script if any command fails
(cd Python && make)
#Check if slender is already present:
if grep "SLENDER_PYTHON" ~/.bashrc; then echo "Slender body already present in bashrc"; exit 0; fi
#If SLENDER_PYTHON is not present in bashrc append these lines to it
cat <<EOF >> ~/.bashrc
export SLENDER_ROOT=$(pwd)
export SLENDER_PYTHON=\$SLENDER_ROOT/Python/Dependencies
export PYTHONPATH=\$SLENDER_PYTHON/BatchedNBodyRPY:\$SLENDER_PYTHON/UAMMD_PSE_Python:\$SLENDER_PYTHON/NeighborSearch:\$SLENDER_PYTHON/:\$PYTHONPATH
EOF
