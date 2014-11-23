#ifndef RAYTRACER_SYSTEM_H
#define RAYTRACER_SYSTEM_H

#include "../include/Config.hpp"

#include <GL/gl.h>

typedef struct RayTracingUnit_t{
	struct cudaGraphicsResource* vertexBuffer;
	struct cudaGraphicsResource* solarCoverageBuffer;
	bool isStar;
} RayTracingUnit;

extern "C"{
	/**
	* Adding a body's vertex and color buffer to the ray traycing system
	**/
	void addBodyToRayTracer(GLuint vertexBuffer, GLuint solarCoverageBuffer, bool isStar, Config *config);
	
	/**
	* Simulates the rays for all the bodies and the stars
	**/
	void rayTracerSimulateRaysOne(int starIndex, double x1, double y1, double z1, int bodyIndex, double x2, double y2, double z2, int numVertices, double *mat);
	
	/**
	* Simulates the rays for all the bodies and the stars
	**/
	void rayTracerSimulateRaysTwo(int sourceIndex, double x1, double y1, double z1, int sourceVertices, int bodyIndex, double x2, double y2, double z2, int bodyVertices, double *bodyMat, double *sourceMat, float intensity);
	
	/**
	* Simulates the rays for all the bodies and the stars
	**/
	void rayTracerUnillunimate(int index, int numBodyVertices);
	
	/**
	* Simulates the rays for all the bodies and the stars
	**/
	void rayTracerIllunimate(int index, int numBodyVertices);
	
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
