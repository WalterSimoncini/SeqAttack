# Disk size should be 50 GB

# Install python3
sudo apt update
sudo apt install python-is-python3 -y

# Install GCC
sudo apt install build-essential -y

# Install anaconda
# Remember to provide the install directory as /home/walter/anaconda3
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
sh Anaconda3-2020.11-Linux-x86_64.sh -b

# Initialize conda
/home/walter/anaconda3/bin/conda init

# Reload shell
source ~/.bashrc

# Create python 3.6 environment
conda create -n thesis python=3.6 -y
conda activate thesis

# Install requirements
# We are not using a requirements.txt file because it causes conflicts
pip install torch==1.7.0
pip install textattack==0.2.15
pip install tensorflow==2.4.2 tensorflow-hub
pip install pygit2 python-Levenshtein
pip install seqeval
pip install openattack