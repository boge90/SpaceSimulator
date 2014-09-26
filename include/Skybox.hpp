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
		
		// Side1
		GLuint tex1;
		BMP *bmp1;
		GLuint tex2;
		BMP *bmp2;
		GLuint tex3;
		BMP *bmp3;
		GLuint tex4;
		BMP *bmp4;
		GLuint tex5;
		BMP *bmp5;
		GLuint tex6;
		BMP *bmp6;
		
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
		void render(const GLfloat *mvp, glm::dvec3 position, glm::dvec3 direction, glm::dvec3 up);
};

#endif
