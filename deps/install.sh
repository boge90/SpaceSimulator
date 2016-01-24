echo 'Installing dependencies'
sudo apt-get install libglew-dev
sudo apt-get install cmake
sudo apt-get install libxrandr-dev

echo 'Installing GLFW3'
unzip glfw-3.0.4.zip
cd glfw-3.0.4
cmake CMakeLists.txt
cd src
make
sudo make install
cd ../include/GLFW
sudo mkdir /usr/include/GLFW
sudo cp glfw3.h /usr/include/GLFW/
