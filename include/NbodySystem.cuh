#ifndef NBODY_SYSTEM_H
#define NBODY_SYSTEM_H

#include <GL/gl.h>

extern "C"{
	/**
	*
	**/
	void initializeNbodySystem(void);

	/**
	* Function for initializing the NBody system running on the GPU
	**/
	void addBodyVertexBuffer(GLuint buffer);
	
	/**
	* Functions for translating all vertices for a body given by transformation matrix
	**/
	void moveBody(int bodyIndex, int numVertices, double *translation, double cx, double cy, double cz, double radius);
}

#endif
