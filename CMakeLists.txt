cmake_minimum_required(VERSION 3.22)
project(kdl_kinematics_2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclpy REQUIRED)
find_package(random_numbers REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_kinematics REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(tf2_kdl REQUIRED)
find_package(class_loader REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(generate_parameter_library REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(ament_cmake REQUIRED)

# Include directories
include_directories(
  include
  ${pluginlib_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
)

# Declare a C++ library
add_library(kdl_kinematics_2 SHARED
  src/kdl_kinematics_2.cpp
  src/chainiksolver_vel_mimic_svd.cpp
)


# Link the dependencies
ament_target_dependencies(kdl_kinematics_2
  pluginlib
  rclcpp
  rclpy
  random_numbers
  moveit_msgs
  moveit_core
  moveit_kinematics
  kdl_parser
  tf2_kdl
  class_loader
  geometry_msgs
  generate_parameter_library
  moveit_ros_planning
  orocos_kdl
  Eigen3
)

##################################
# Finds Boost Components
#include(ConfigExtras.cmake)

generate_parameter_library(
  kdl_kinematics_2_parameters # cmake target name for the parameter library
  src/kdl_kinematics_2_parameters.yaml # path to input yaml file
)

target_link_libraries(kdl_kinematics_2 kdl_kinematics_2_parameters)

# prevent pluginlib from using boost
target_compile_definitions(kdl_kinematics_2
                           PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")
# Causes the visibility macros to use dllexport rather than dllimport, which is
# appropriate when building the dll but not consuming it.
target_compile_definitions(kdl_kinematics_2
                           PRIVATE "MOVEIT_KDL_KINEMATICS_PLUGIN_BUILDING_DLL")
##################################

# Install the library
install(TARGETS kdl_kinematics_2
#  EXPORT kdl_kinematics_2Targets
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install include directories
install(DIRECTORY include/
  DESTINATION include/
)

install(FILES kdl_kinematics_2_description.xml
  DESTINATION share/${PROJECT_NAME}
)

pluginlib_export_plugin_description_file(${PROJECT_NAME} kdl_kinematics_2_description.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Export package dependencies
#################################
#ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
#################################

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(pluginlib rclcpp rclpy random_numbers moveit_msgs moveit_core moveit_kinematics kdl_parser tf2_kdl class_loader geometry_msgs generate_parameter_library  moveit_ros_planning orocos_kdl Eigen3)
ament_package()
