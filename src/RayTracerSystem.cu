#include "../include/RayTracerSystem.cuh"
#include <stdio.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// CUDA resources
static std::vector<RayTracingUnit> resources; 

// CUDA function prototypes
void __global__ simulateRays(double3 bc, double3 sc, int numBodyVertices, int numStarVertices, double3 *bodyVertices, double3 *starVertices, float3 *bodyColors);

void addBodyToRayTracer(GLuint vertexBuffer, GLuint colorBuffer, int numVertices, bool isStar, Config *config){
	// Debug
	if((config->getDebugLevel() & 0x8) == 8){	
		printf("RayTracerSystem.cu\tAdding body to ray tracer system (%d, %d, %d, %d)\n", vertexBuffer, colorBuffer, numVertices, isStar);
	}
	
	// Initializing unit
	struct cudaGraphicsResource *vertexResource;
	cudaGraphicsGLRegisterBuffer(&vertexResource, vertexBuffer, cudaGraphicsRegisterFlagsNone);
	
	struct cudaGraphicsResource *colorResource;
	cudaGraphicsGLRegisterBuffer(&colorResource, colorBuffer, cudaGraphicsRegisterFlagsNone);
	
	RayTracingUnit unit;
	unit.vertexBuffer = vertexResource;
	unit.colorBuffer = colorResource;
	unit.numVertices = numVertices;
	unit.isStar = isStar;
	
	// Adding body to body list
	resources.push_back(unit);		
}

void rayTracerSimulateRays(int starIndex, double x1, double y1, double z1, int bodyIndex, double x2, double y2, double z2){
	// Local vars
	double3 *starVertices = 0;
	double3 *bodyVertices = 0;
	float3 *bodyColors = 0;
	size_t num_bytes_starVertices;
	size_t num_bytes_bodyVertices;
	size_t num_bytes_bodyColors;
	
	// Getting the arrays
	cudaGraphicsResourceGetMappedPointer((void**)&starVertices, &num_bytes_starVertices, resources[starIndex].vertexBuffer);
	cudaGraphicsResourceGetMappedPointer((void**)&bodyVertices, &num_bytes_bodyVertices, resources[bodyIndex].vertexBuffer);
	cudaGraphicsResourceGetMappedPointer((void**)&bodyColors, &num_bytes_bodyColors, resources[bodyIndex].colorBuffer);
	
	int numBodyVertices = resources[bodyIndex].numVertices;
	int numStarVertices = resources[starIndex].numVertices;
	
	dim3 grid((numBodyVertices/512) + 1);
	dim3 block(512);
	
	simulateRays<<<grid, block>>>(make_double3(x2,y2,z2), make_double3(x1,y1,z1), numBodyVertices, numStarVertices, bodyVertices, starVertices, bodyColors);
}

void prepareRaySimulation(void){
	for(size_t i=0; i<resources.size(); i++){
		RayTracingUnit u = resources[i];
		
		// Mapping vertex buffer
		cudaGraphicsMapResources(1, &resources[i].vertexBuffer);	
		
		if(!u.isStar){
			cudaGraphicsMapResources(1, &resources[i].colorBuffer);	
		}
	}
}


void finalizeRaySimulation(void){
	for(size_t i=0; i<resources.size(); i++){
		RayTracingUnit u = resources[i];
		
		// Mapping vertex buffer
		cudaGraphicsUnmapResources(1, &resources[i].vertexBuffer);	
		
		if(!u.isStar){
			cudaGraphicsUnmapResources(1, &resources[i].colorBuffer);	
		}
	}
}


void __global__ simulateRays(double3 bc, double3 sc, int numBodyVertices, int numStarVertices, double3 *bodyVertices, double3 *starVertices, float3 *bodyColors){
	// Global thread index
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Vertex data
	float lightIntensity = 0.f;
	double3 bodyVertex = bodyVertices[index];
	double3 bodyNormal;
	bodyNormal.x = bodyVertex.x - bc.x;
	bodyNormal.y = bodyVertex.y - bc.y;
	bodyNormal.z = bodyVertex.z - bc.z;
	
	// Will spawn some extra threads which must be terminated
	if(index >= numBodyVertices){return;}
	
	// Checking star vertices
	for(int i=0; i<numStarVertices; i++){
		double3 starVertex = starVertices[i];
		
		double planEq = bodyNormal.x * (starVertex.x - bodyVertex.x) + bodyNormal.y * (starVertex.y - bodyVertex.y) + bodyNormal.z * (starVertex.z - bodyVertex.z);
		
		if(planEq > 0){
			lightIntensity += 1.f/numStarVertices;
		}
	}
	
	// Setting vertex color
	bodyColors[index].x = lightIntensity;
	bodyColors[index].y = lightIntensity;
	bodyColors[index].z = lightIntensity;
}
