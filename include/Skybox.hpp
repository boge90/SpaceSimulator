#ifndef SKYBOX_H
#define SKYBOX_H

#include "../include/Renderable.hpp"
#include "../include/Shader.hpp"
#include "../include/Config.hpp"

class Skybox: public Renderable{
	private:
		// Rendering
		Shader *shader;
		
		// Misc
		size_t debugLevel;
	public:
		/**
		* Constructs the skybox object
		**/
		Skybox(Config *config);
		
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
