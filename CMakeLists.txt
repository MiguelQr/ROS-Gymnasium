cmake_minimum_required(VERSION 3.8)
project(ros_gymnasium)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)


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

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_gymnasium_api.py")
    find_package(ament_cmake_pytest REQUIRED)
    ament_add_pytest_test(test_gymnasium_api test/test_gymnasium_api.py TIMEOUT 300)
  endif()
endif()

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/go_to_pose_camera
  DESTINATION lib/${PROJECT_NAME}
)

# scripts/multi_agent_training is work in progress and intentionally not installed

install(
  DIRECTORY launch config goals_and_poses worlds models
  DESTINATION share/${PROJECT_NAME}
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/ros_gymnasium.dsv.in")

ament_package()
