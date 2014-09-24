#ifndef SHADER_H
#define SHADER_H

#include "../include/Config.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

#include <stdlib.h>
#include <GL/glew.h>

// Include GLM
#include <glm/glm.hpp>
using namespace std;

class Shader{
	private:
		// OpenGl
		GLuint shaderId;
		
		// Misc
		size_t debugLevel;
		
		/**
		* Method for loading and compiling shaders
		**/
		GLuint loadShaders(const char *vertex_file_path, const char *fragment_file_path);
	public:
		/**
		* Loads the shaders specified from arguments and compiles them
		**/
		Shader(const char *vertexShader, const char *fragmentShader, Config *config);
		
		/**
		* Finalizes the shader
		**/
		~Shader(void);
		
		/**
		* Binds this shader
		**/
		void bind(void);
		
		/**
		* Returns the identifier for this shader
		**/
		GLuint getID(void);
};

#endif
