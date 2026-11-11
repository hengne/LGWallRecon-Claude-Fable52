# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
cmake_minimum_required(VERSION 3.16)
project(lgsim CXX)
find_package(Geant4 REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
add_executable(lgsim lgsim.cc)
target_link_libraries(lgsim ${Geant4_LIBRARIES})
target_include_directories(lgsim PRIVATE ${Geant4_INCLUDE_DIRS})
