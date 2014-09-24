#ifndef OPEN_GL_HELPER
#define OPEN_GL_HELPER

#include "../include/Config.hpp"

#include <GL/glew.h> // Must be included before gl.h
#include <GLFW/glfw3.h>
#include <GL/gl.h>

class OpenGlHelper{
	public:
		static GLFWwindow* init(Config *config);
};

#endif
