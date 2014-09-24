#ifndef BODY_ROTATOR_SYSTEM_H
#define BODY_ROTATOR_SYSTEM_H

#include "../include/Config.hpp"
#include <GL/gl.h>

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
	void addBodyToRotationSystem(GLuint buffer, Config *config);
	
	/**
	*
	**/
	void rotateBody(int bodyNum, int numVertices, double *matrix, Config *config);
	
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
