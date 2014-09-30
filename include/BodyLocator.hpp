#ifndef BODY_LOCATOR_H
#define BODY_LOCATOR_H

#include "../include/Renderable.hpp"
#include "../include/Config.hpp"
#include "../include/Shader.hpp"
#include "../include/Body.hpp"

#include <vector>

class BodyLocator: public Renderable{
	private:
		// Debug
		Config *config;
		size_t debugLevel;
	
		// Data
		std::vector<Body*> *bodies;
		
		// Visualization
		Shader *shader;
		bool active;
		int bodyIndex;
		GLuint vertexBuffer;
		GLuint colorBuffer;
	public:
		/**
		*
		**/
		BodyLocator(std::vector<Body*> *bodies, Config *config);
		
		/**
		*
		**/
		~BodyLocator();
	
		/**
		*
		**/
		void render(glm::mat4 *vp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up);
		
		/**
		*
		**/
		void locateBody(size_t bodyIndex);
		
		/**
		*
		**/
		void setActive(bool active);
};

#endif
