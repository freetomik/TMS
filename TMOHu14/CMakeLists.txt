CMAKE_MINIMUM_REQUIRED(VERSION 2.4.8)

SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_INCLUDE_CURRENT_DIR TRUE)

IF(UNIX AND NOT LINUX_SET)
 ADD_DEFINITIONS(-D LINUX)
ENDIF(UNIX AND NOT LINUX_SET)


FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_VERSION VERSION_LESS "2.4.9")
  message(FATAL_ERROR "ERROR: Can't find OpenCV 2.4.9+")
ELSE(OpenCV_VERSION VERSION_LESS "2.4.9") 
	MESSAGE("OpenCV found OK")
ENDIF()

INCLUDE_DIRECTORIES(
${OpenCV_INCLUDE_DIRS}
../tmolib/
.
)

LINK_LIBRARIES (
${OpenCV_LIBS}
tmo


)



SET(TMO_SOURCES
 TMOHu14.cpp
 TMOPlugin.cpp
)


SET(TMOWARD_HEADERS
 TMOHu14.h
 TMOPlugin.h
)


add_library( Hu14 SHARED ${TMO_SOURCES} )
 add_custom_command( TARGET Hu14 POST_BUILD 
 	COMMAND cp -f libHu14.so ../TMOCmd/Hu14.tml
 	COMMAND cp -f libHu14.so ../TMOgui/Hu14.tml 
 	COMMAND cp -f libHu14.so ../Hu14.tml 
 #	COMMAND rm -f libHu14.so 
  	)


