cmake_minimum_required(VERSION 3.5)

# Define the project name
project(Project)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source files
set(SOURCES
    src/main.cpp
    src/value.cpp
)

# Enable debug symbols and GDB-friendly settings
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# Add the executable
add_executable(runme ${SOURCES})

