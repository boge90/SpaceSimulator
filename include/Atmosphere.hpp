#ifndef ATMOSPHERE_H
#define ATMOSPHERE_H

class Atmosphere;

#include "../include/Body.hpp"
#include "../include/Config.hpp"
#include "../include/Shader.hpp"
#include "../include/Renderable.hpp"

class Atmosphere: public Renderable{
	private:
		// Debug
		size_t debugLevel;
		Config *config;
		
		// Data
		Body *body;
		
		// Render
		Shader *shader;
	public:
		/**
		*
		**/
		Atmosphere(Body *body, Config *config);
		
		/**
		*
		**/
		~Atmosphere();
		
		/**
		* Uses OpenGL to draw this atmosphere with the current shader
		**/
		void render(glm::mat4 *vp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up);
};

#endif
