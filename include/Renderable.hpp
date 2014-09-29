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
		virtual void render(glm::mat4 *vp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up) = 0;
};

#endif
