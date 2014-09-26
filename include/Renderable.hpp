#ifndef RENDERABLE_H
#define RENDERABLE_H

#include "../include/common.hpp"

class Renderable{
	public:			
		/**
		* Deconstructor
		**/
		virtual ~Renderable(void){};
		
		/**
		* Uses OpenGL to draw this body with the current shader
		**/
		virtual void render(const GLfloat *mvp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up) = 0;
};

#endif
