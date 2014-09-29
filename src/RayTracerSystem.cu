#include "../include/RayTracerSystem.cuh"
#include <stdio.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// CUDA Memory
__constant__ double cuda_translation_matrix[16];

// CUDA resources
static std::vector<RayTracingUnit> resources; 

// CUDA function prototypes
void __global__ simulateRays(double3 bc, double3 sc, int numBodyVertices, double3 *bodyVertices, float *solarCoverageBuffer);

void addBodyToRayTracer(GLuint vertexBuffer, GLuint solarCoverageBuffer, int numVertices, bool isStar, Config *config){
	// Debug
	if((config->getDebugLevel() & 0x8) == 8){	
		printf("RayTracerSystem.cu\tAdding body to ray tracer system (%d, %d, %d, %d)\n", vertexBuffer, solarCoverageBuffer, numVertices, isStar);
	}
	
	// Initializing unit
	struct cudaGraphicsResource *vertexResource;
	cudaGraphicsGLRegisterBuffer(&vertexResource, vertexBuffer, cudaGraphicsRegisterFlagsNone);
	
	struct cudaGraphicsResource *solarCoverageResource;
	cudaGraphicsGLRegisterBuffer(&solarCoverageResource, solarCoverageBuffer, cudaGraphicsRegisterFlagsNone);
	
	RayTracingUnit unit;
	unit.solarCoverageBuffer = solarCoverageResource;
	unit.vertexBuffer = vertexResource;
	unit.numVertices = numVertices;
	unit.isStar = isStar;
	
	// Adding body to body list
	resources.push_back(unit);
}

void rayTracerSimulateRays(int starIndex, double x1, double y1, double z1, int bodyIndex, double x2, double y2, double z2, double *mat){
	// Local vars
	double3 *bodyVertices = 0;
	float *solarCoverage = 0;
	size_t num_bytes_bodyVertices;
	size_t num_bytes_solarCoverage;
	
	// Transferring the translation matrix
	cudaMemcpyToSymbol(cuda_translation_matrix, mat, 4*4*sizeof(double), 0, cudaMemcpyHostToDevice);
	
	// Getting the arrays
	cudaGraphicsResourceGetMappedPointer((void**)&bodyVertices, &num_bytes_bodyVertices, resources[bodyIndex].vertexBuffer);
	cudaGraphicsResourceGetMappedPointer((void**)&solarCoverage, &num_bytes_solarCoverage, resources[bodyIndex].solarCoverageBuffer);
	
	int numBodyVertices = resources[bodyIndex].numVertices;
	
	dim3 grid((numBodyVertices/512) + 1);
	dim3 block(512);
	
	simulateRays<<<grid, block>>>(make_double3(x2,y2,z2), make_double3(x1,y1,z1), numBodyVertices, bodyVertices, solarCoverage);
}

void prepareRaySimulation(void){
	for(size_t i=0; i<resources.size(); i++){
		RayTracingUnit u = resources[i];
		
		// Mapping vertex buffer
		cudaGraphicsMapResources(1, &resources[i].vertexBuffer);	
		
		if(!u.isStar){
			cudaGraphicsMapResources(1, &resources[i].solarCoverageBuffer);	
		}
	}
}


void finalizeRaySimulation(void){
	for(size_t i=0; i<resources.size(); i++){
		RayTracingUnit u = resources[i];
		
		// Mapping vertex buffer
		cudaGraphicsUnmapResources(1, &resources[i].vertexBuffer);	
		
		if(!u.isStar){
			cudaGraphicsUnmapResources(1, &resources[i].solarCoverageBuffer);	
		}
	}
}


void __global__ simulateRays(double3 bc, double3 sc, int numBodyVertices, double3 *bodyVertices, float *solarCoverageBuffer){
	// Global thread index
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// Will spawn some extra threads which must be terminated
	if(index >= numBodyVertices){return;}

	// Vertex data
	float lightIntensity = 0.f;
	double3 bodyNormal = bodyVertices[index];
	
	double vx = cuda_translation_matrix[0]*bodyNormal.x + cuda_translation_matrix[4]*bodyNormal.y + cuda_translation_matrix[8]*bodyNormal.z + cuda_translation_matrix[12];
	double vy = cuda_translation_matrix[1]*bodyNormal.x + cuda_translation_matrix[5]*bodyNormal.y + cuda_translation_matrix[9]*bodyNormal.z + cuda_translation_matrix[13];
	double vz = cuda_translation_matrix[2]*bodyNormal.x + cuda_translation_matrix[6]*bodyNormal.y + cuda_translation_matrix[10]*bodyNormal.z + cuda_translation_matrix[14];
	
	bodyNormal.x = vx-bc.x;
	bodyNormal.y = vy-bc.y;
	bodyNormal.z = vz-bc.z;
	
	double planEq = bodyNormal.x * (sc.x - vx) + bodyNormal.y * (sc.y - vy) + bodyNormal.z * (sc.z - vz);
	
	if(planEq > 0){
		lightIntensity = 1.f;
	}
	
	// Setting vertex color
	solarCoverageBuffer[index] = lightIntensity;
}
