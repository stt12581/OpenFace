cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "-std=c++0x")

add_executable(TrainModel svm.h svm.cpp TrainModel.cpp)
set_property(TARGET TrainModel PROPERTY CXX_STANDARD 11)
set_property(TARGET TrainModel PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(TrainModel pca armadillo)
