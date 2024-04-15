module load GCC/6.4.0-2.28
module load Python/3.6.4

# Create new virtual environment
python -m venv ./venv

# Modules for pip installs and compiles
module load GCCcore/6.4.0
module load CMake/3.9.5

source venv/bin/activate

# Install python dependencies from requirements.txt
python -m pip install pip==21.3.1
pip install -r requirements.txt

deactivate
module purge

cd venv/
mkdir dependencies
cd dependencies

# Compile and install glibc 2.23
wget http://ftp.gnu.org/gnu/glibc/glibc-2.23.tar.gz
tar zxvf glibc-2.23.tar.gz
cd glibc-2.23
mkdir build
cd build/
../configure --disable-sanity-checks
make -j4 CFLAGS="-O2 -g -Wno-error"
make install DESTDIR=../../glibc_install
cd ../../
cp glibc_install/usr/local/lib/libm-2.23.so glibc_install/usr/local/lib/libm.so.6

# Compile and install MKL
git clone https://github.com/01org/mkl-dnn.git
cd mkl-dnn
git checkout rls-v0.21
cd scripts && ./prepare_mkl.sh && cd ..
mkdir ../mkl
cp -r external/mklml_lnx_2019.0.5.20190502/lib/ ../mkl/lib
# mkdir -p build && cd build && cmake .. && make
# make install DESTDIR=../../mkl

# Add new dependencies to LD_LIBRARY_PATH
cd ../../
chmod 777 bin/activate
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/dependencies/mkl/lib" >> "bin/activate"
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/dependencies/glibc_install/usr/local/lib" >> "bin/activate"