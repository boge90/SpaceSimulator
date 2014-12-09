#ifndef NBODY_SYSTEM_H
#define NBODY_SYSTEM_H

#include "../include/Config.hpp"

#include <GL/gl.h>

extern "C"{
	/**
	*
	**/
	void initializeNbodySystem(double G, double dt, double *positions, double *velocities, double *mass, size_t numBodies, Config *config);
	
	/**
	* Functions for translating all vertices for a body given by transformation matrix
	**/
	void update(double *newPositions);
}

#endif
