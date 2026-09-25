cmake_minimum_required(VERSION 3.16)
project(Condor LANGUAGES CXX)
set(PROJECT_VERSION "0.1.0-dev")

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24")
    cmake_policy(SET CMP0135 NEW)
endif()

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Packages
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Kokkos REQUIRED)
find_package(Stork REQUIRED)
find_package(nlohmann_json REQUIRED)

# Gather all .cpp files in the include/ directory
file(GLOB_RECURSE SRC_FILES include/*.cpp)

# Gather all .h/.hpp files in the include/ directory
file(GLOB_RECURSE INCLUDE_FILES include/*.hpp)

# Core library
add_library(Condor ${SRC_FILES})

# Link the dependencies to the Condor library
target_link_libraries(Condor
    PUBLIC
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    Kokkos::kokkos
    Stork::Stork
    nlohmann_json::nlohmann_json
)

# Include directories for Condor
target_include_directories(Condor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Apply the definitions
target_compile_definitions(Condor PUBLIC
    ${CONDOR_COMPILE_DEFINITIONS}
)

# Set the output directory for the executables during build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps)

# Add executables for each file in apps/
file(GLOB APP_SOURCES "apps/*.cpp")
foreach(app_source ${APP_SOURCES})
    get_filename_component(app_name ${app_source} NAME_WE)
    add_executable(${app_name} ${app_source})
    target_link_libraries(${app_name} PRIVATE Condor)
endforeach()

# Install targets, headers, and package config (rest of your install commands remain the same)
install(TARGETS Condor
    EXPORT CondorConfig
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Package configuration for find_package() support (rest is the same)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CondorConfigVersion.cmake"
    VERSION 1.0
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/CondorConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CondorConfig.cmake"
    INSTALL_DESTINATION lib/cmake/Condor
)

install(EXPORT CondorConfig
    DESTINATION lib/cmake/Condor
    NAMESPACE Condor::
)

install(EXPORT CondorConfig
    FILE CondorTargets.cmake
    NAMESPACE Condor::
    DESTINATION lib/cmake/Condor
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CondorConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CondorConfigVersion.cmake"
    DESTINATION lib/cmake/Condor
)
