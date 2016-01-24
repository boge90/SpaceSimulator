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
		GLuint programId;
		
		// Misc
		size_t debugLevel;
	public:
		/**
		* Loads the shaders specified from arguments and compiles them
		**/
		Shader(Config *config);
		
		/**
		* Finalizes the shader
		**/
		~Shader(void);
		
		/**
		* Method for loading and compiling shaders
		**/
		void addShader(const char *path, GLenum shaderType);
		
		/**
		*
		**/
		void link(void);
		
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
