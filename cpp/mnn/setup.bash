#! /bin/bash
sudo apt install -y libyaml-cpp0.6
mkdir build
cd build
cmake ..
make -j8
cd ..
echo "\n"
./build/mnn_det -?
