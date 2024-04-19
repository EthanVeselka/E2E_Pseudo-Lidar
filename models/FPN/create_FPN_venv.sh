mkdir venv
cd venv/
mkdir dependencies
cd dependencies/

# Load modules for GCC build/install
module load GCC/8.3.0
module load MPC/1.1.0
module load GMP/6.1.2
module load MPFR/4.0.2

# Download and make GCC 5.3
wget https://ftp.gnu.org/gnu/gcc/gcc-5.3.0/gcc-5.3.0.tar.gz
tar -xzf gcc-5.3.0.tar.gz
cd gcc-5.3.0
./configure --disable-multilib
make
make
make install DESTDIR=$(pwd)/gcc

module purge

# Load Python 3.6.4
export PATH=/sw/eb/sw/Python/3.6.4-golf-2018a/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/eb/sw/Python/3.6.4-golf-2018a/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/eb/sw/Python/3.6.4-golf-2018a/lib/python3.6/site-packages/numpy-1.14.0-py3.6-linux-x86_64.egg/numpy/core/lib

# Load GCC 5.3
export PATH=$(pwd)/venv/dependencies/gcc-5.3.0/gcc/usr/local/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/venv/dependencies/gcc-5.3.0/gcc/usr/local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/venv/dependencies/gcc-5.3.0/gcc/usr/local/lib64

# Create new virtual environment
python -m venv ./venv

source venv/bin/activate

# Install python dependencies from requirements.txt
python -m pip install pip==21.3.1
pip install -r fpn_pip_requirements.txt

deactivate

unset PATH
unset LD_LIBRARY_PATH

export PATH=/usr/lib64/qt-3.3/bin:/sw/local/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/lpp/mmfs/bin