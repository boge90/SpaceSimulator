#ifndef SKYBOX_H
#define SKYBOX_H

#include "../include/Renderable.hpp"
#include "../include/Shader.hpp"

class Skybox: public Renderable{
	private:
		// Rendering
		Shader *shader;
	public:
		/**
		* Constructs the skybox object
		**/
		Skybox();
		
		/**
		* Finalizes the skybox object
		**/
		~Skybox();
	
		/**
		* Renders the skybox
		**/
		void render(const GLfloat *mvp);
};

#endif
