cmake_minimum_required(VERSION 2.4)
project( pf )
find_package( OpenCV REQUIRED )
add_executable( pf pf.cpp )
target_link_libraries( pf ${OpenCV_LIBS} )
