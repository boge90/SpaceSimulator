#ifndef BODY_ROTATOR_SYSTEM_H
#define BODY_ROTATOR_SYSTEM_H

#include "../include/Config.hpp"
#include <GL/gl.h>

typedef struct BodyRotationUnit_t{
	struct cudaGraphicsResource* vertexResource;
	int numVertices;
} BodyRotationUnit;

extern "C"{
	/**
	*
	**/
	void initializeBodyRotatorSystem(Config *config);
	
	/**
	*
	**/
	void finalizeBodyRotatorSystem(Config *config);
	
	/**
	*
	**/
	void addBodyToRotationSystem(GLuint vertexBuffer, int numVertices, Config *config);
	
	/**
	*
	**/
	void rotateBody(int bodyNum, double *matrix, Config *config);
	
	/**
	*
	**/
	void prepareBodyRotation(Config *config);
	
	/**
	*
	**/
	void endBodyRotation(Config *config);
}

#endif
