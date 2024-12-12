# ==================================================================================================
# @brief Enable SPDLOG support.
# 
# @note Several parameters should be set before including this file:
#       - [ENV]{SPDLOG_HOME}/[ENV]{SPDLOG_DIR}:
#             Path to spdlog libaray installation path.
# ==================================================================================================

include(${CMAKE_CURRENT_LIST_DIR}/../utils/common.cmake)

try_get_value(SPDLOG_HOME HINTS SPDLOG_HOME SPDLOG_DIR)
if (NOT SPDLOG_HOME_FOUND)
    log_error("[ENV]{SPDLOG_HOME}/[ENV]{SPDLOG_DIR} not set.")
endif()

set(SPDLOG_CMAKE_PREFIX_PATH "${SPDLOG_HOME}/lib/cmake")
list(APPEND CMAKE_PREFIX_PATH ${SPDLOG_CMAKE_PREFIX_PATH})

find_package(spdlog REQUIRED)