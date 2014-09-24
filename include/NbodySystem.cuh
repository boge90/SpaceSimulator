#ifndef NBODY_SYSTEM_H
#define NBODY_SYSTEM_H

#include "../include/Config.hpp"

#include <GL/gl.h>

extern "C"{
	/**
	*
	**/
	void initializeNbodySystem(Config *config);

	/**
	* Function for initializing the NBody system running on the GPU
	**/
	void addBodyVertexBuffer(GLuint buffer, Config *config);
	
	/**
	* Functions for translating all vertices for a body given by transformation matrix
	**/
	void moveBody(int bodyIndex, int numVertices, double *translation);
}

#endif
