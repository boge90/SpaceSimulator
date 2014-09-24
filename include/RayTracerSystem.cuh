#ifndef RAYTRACER_SYSTEM_H
#define RAYTRACER_SYSTEM_H

#include "../include/Config.hpp"

#include <GL/gl.h>

typedef struct RayTracingUnit_t{
	struct cudaGraphicsResource* vertexBuffer;
	struct cudaGraphicsResource* colorBuffer;
	int numVertices;
	bool isStar;
} RayTracingUnit;

extern "C"{
	/**
	* Adding a body's vertex and color buffer to the ray traycing system
	**/
	void addBodyToRayTracer(GLuint vertexBuffer, GLuint colorBuffer, int numVertices, bool isStar, Config *config);
	
	/**
	* Simulates the rays for all the bodies and the stars
	**/
	void rayTracerSimulateRays(int starIndex, double x1, double y1, double z1, int bodyIndex, double x2, double y2, double z2);
	
	/**
	* Prepares the buffers, enabling them for the calculation
	**/
	void prepareRaySimulation(void);
	
	/**
	* Finalizes the simulation step, enabling the buffers for rendering
	**/
	void finalizeRaySimulation(void);
}

#endif
