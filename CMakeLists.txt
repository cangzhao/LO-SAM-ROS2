cmake_minimum_required(VERSION 3.5.1)
project(lo_sam)

# 必须先找到ament_cmake
find_package(ament_cmake REQUIRED)

# 现代C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# 设置rpath
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,'\$ORIGIN/'")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'\$ORIGIN/'")


# 预定义宏，替代 add_definitions
add_compile_definitions(ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

# 找第三方库
find_package(Boost REQUIRED COMPONENTS thread)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)

# ROS 2核心包和消息
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

# tf2相关
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# 生成消息
set(msg_files 
    "msg/CloudInfo.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
    ${msg_files}
    DEPENDENCIES std_msgs geometry_msgs nav_msgs sensor_msgs tf2_geometry_msgs visualization_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# 编译可执行文件 - 下面示例中，替换掉原来的catkin_LIBRARIES为相应的ament依赖和link库

# feature_extract
add_executable(feature_extract src/feature_extract.cpp)

target_include_directories(feature_extract PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(feature_extract  
  ${PCL_LIBRARIES}
)
ament_target_dependencies(feature_extract
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
  sensor_msgs
  pcl_conversions
  tf2_geometry_msgs
  visualization_msgs
)
add_dependencies(feature_extract ${cpp_typesupport_target})


# laserOdometry
add_executable(laserOdometry src/laserOdometry.cpp)

target_include_directories(laserOdometry PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp/${PROJECT_NAME}> 
)

target_link_libraries(laserOdometry
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  ${cpp_typesupport_target}
)
ament_target_dependencies(laserOdometry
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
  sensor_msgs
  pcl_conversions
  tf2_geometry_msgs
  visualization_msgs
)

# map_optimization
add_executable(map_optimization src/map_optimization.cpp)
target_include_directories(map_optimization PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp/${PROJECT_NAME}>
)
target_compile_options(map_optimization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(map_optimization
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  gtsam
  ${cpp_typesupport_target}
  ${OpenMP_CXX_FLAGS}
)
ament_target_dependencies(map_optimization
  rclcpp
  geometry_msgs
  nav_msgs
  std_msgs
  sensor_msgs
  pcl_conversions
  tf2_geometry_msgs
  visualization_msgs
)
# 安装规则（示例）
install(TARGETS
  feature_extract
  laserOdometry
  map_optimization
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)
install(DIRECTORY rviz_cfg/
  DESTINATION share/${PROJECT_NAME}/rviz_cfg
  FILES_MATCHING PATTERN "*.rviz"
)
# 生成接口和包信息
ament_package()
