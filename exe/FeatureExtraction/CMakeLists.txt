cmake_minimum_required(VERSION 2.8)

#TBB library
include_directories(${TBB_ROOT_DIR}/include)
include_directories(${Boost_INCLUDE_DIR})

include_directories(~/libpca/include)
link_directories(~/libpca/lib)

add_executable(FeatureExtraction svm.h svm.cpp FeatureExtraction.cpp)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)		

target_link_libraries(FeatureExtraction LandmarkDetector)
target_link_libraries(FeatureExtraction FaceAnalyser)
target_link_libraries(FeatureExtraction dlib)
target_link_libraries(FeatureExtraction ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})
target_link_libraries(FeatureExtraction pca)

#add_executable(TrainModel TrainModel.cpp svm.cpp)
#link_directories(~/libpca/lib)
#include_directories(~/libpca/include)
#target_link_libraries(TrainModel pca)
