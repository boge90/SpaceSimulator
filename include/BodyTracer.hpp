#ifndef BODY_TRACER_H
#define BODY_TRACER_H

#include "../include/common.hpp"
#include "../include/Body.hpp"
#include <vector>

class BodyTracer: public Renderable{
	private:
		// Calculation data
		size_t MAX_VERTICES;
		double dt;
		double G;
		std::vector<Body*> *bodies;
		
		// OpenGL
		Shader *shader;
		size_t bodyNum;
		bool active;
		size_t numVertices;
		GLuint vertexBuffer;
		GLuint colorBuffer;
	public:
		/**
		*
		**/
		BodyTracer(std::vector<Body*> *bodies, double dt);
		
		/**
		*
		**/
		~BodyTracer(void);
		
		/**
		* Uses OpenGL to draw the tracks of the bodies
		**/
		void render(const GLfloat *mvp);
		
		/**
		* Enable / Disables the rendering
		**/
		void toogle(void);
		
		/**
		* Updates the visualization buffer base on the new body positions
		**/
		void calculateFuturePath(size_t bodyNum);
};

#endif
