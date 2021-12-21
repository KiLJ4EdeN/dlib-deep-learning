# dlib cmake example
# install devtoolset 7 on centos 7
# comment these on ubuntu, gcc 4.9+ required
set(CMAKE_C_COMPILER /opt/rh/devtoolset-7/root/usr/bin/gcc)
set(CMAKE_CXX_COMPILER /opt/rh/devtoolset-7/root/usr/bin/g++)


cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(example)


# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.
add_subdirectory(dlib dlib_build)

# The next thing we need to do is tell CMake about the code you want to
# compile.  We do this with the add_executable() statement which takes the name
# of the output executable and then a list of .cpp files to compile.
# chain main.cpp to the file being used
add_executable(app main.cpp)
# Finally, you need to tell CMake that this program, "app",
# depends on dlib.  You do that with this statement: 
target_link_libraries(app dlib::dlib)
