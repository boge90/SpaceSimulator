#ifndef SKYBOX_H
#define SKYBOX_H

#include "../include/Renderable.hpp"
#include "../include/Shader.hpp"
#include "../include/Config.hpp"
#include "../include/BMP.hpp"

class Skybox: public Renderable{
	private:
		// Rendering
		Shader *shader;
		
		// Visualization
		GLuint vertexBuffer1;
		GLuint vertexBuffer2;
		GLuint vertexBuffer3;
		GLuint vertexBuffer4;
		GLuint vertexBuffer5;
		GLuint vertexBuffer6;
		GLuint texCordsBuffer;
		
		// Sides
		GLuint tex1;
		GLuint tex2;
		GLuint tex3;
		GLuint tex4;
		GLuint tex5;
		GLuint tex6;
		
		// Rendering data
		float intensity;
		
		// Misc
		Config *config;
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
		void render(glm::mat4 *vp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up);
		
		/**
		*
		**/
		void setIntensity(float intensity);
};

#endif
