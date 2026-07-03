cmake_minimum_required(VERSION 3.5)
project(yolo_trt_ros2)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  yolo_trt_ros2/yolo_detector_node.py
  yolo_trt_ros2/coordinate_projector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
