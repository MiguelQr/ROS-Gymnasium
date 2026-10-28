cmake_minimum_required(VERSION 3.8)
project(ros_gymnasium)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_gymnasium_api.py")
    find_package(ament_cmake_pytest REQUIRED)
    ament_add_pytest_test(test_gymnasium_api test/test_gymnasium_api.py TIMEOUT 300)
  endif()
endif()

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/go_to_pose_camera
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config goals_and_poses worlds models
  DESTINATION share/${PROJECT_NAME}
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/ros_gymnasium.dsv.in")

ament_package()
