#include "../include/OpenGlHelper.hpp"

#include <iostream>

GLFWwindow* OpenGlHelper::init(Config *config){
	if((config->getDebugLevel() & 0x10) == 16){		
		std::cout << "OpenGlHelper.cpp\tInitializing" << std::endl;
	}
	
	// Initialise GLFW
	if(!glfwInit()){
		std::cerr << "Failed to initialize GLFW\n";
		return NULL;
	}else if((config->getDebugLevel() & 0x8) == 8){
		std::cout << "OpenGlHelper.cpp\tInitialized GLFW\n";
	}
	
	// OpenGL hints
	glfwWindowHint(GLFW_SAMPLES, 4); 								// 4x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); 					// We want OpenGL 4.4
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); 	//We don't want the old OpenGL
	
	GLFWwindow *window = glfwCreateWindow(1, 1, "Init", NULL, NULL);
	glfwMakeContextCurrent(window);
	
	// Initialize GLEW
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
    if (GLEW_OK != err){
        std::cout << "OpenGlHelper.cpp\tError: " << glewGetErrorString(err) << "\n";
        return NULL;
    }else if((config->getDebugLevel() & 0x8) == 8){ 
	    std::cout << "OpenGlHelper.cpp\tStatus: Using GLEW " << glewGetString(GLEW_VERSION) << "\n";
    }
    
    int error = glGetError();
    if(error != 0){    
		std::cout << "OpenGlHelper.cpp\tGLEW Init (Error = " << glewGetErrorString(error) << ")\n";
    }
    
	std::cout << "OpenGlHelper.cpp\tGLSL " << glGetString(GL_SHADING_LANGUAGE_VERSION_ARB) << std::endl;
	
	return window;
}
