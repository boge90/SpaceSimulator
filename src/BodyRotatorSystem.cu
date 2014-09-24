#include "../include/BodyRotatorSystem.cuh"
#include <stdio.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// CUDA resources
static std::vector<struct cudaGraphicsResource*> resources;

// CUDA Translation matrix
__constant__ double cuda_translation_matrix[16];

// CUDA kernel
__global__ void simulateBodyRotation(double3 *vertices, int numVertices);

void initializeBodyRotatorSystem(Config *config){
	// Debug
	if((config->getDebugLevel() & 0x10) == 16){	
		printf("BodyRotatorSystem.cu\tInitializing\n");
	}
}

void finalizeBodyRotatorSystem(Config *config){
	// Debug
	if((config->getDebugLevel() & 0x10) == 16){	
		printf("BodyRotatorSystem.cu\tFinalizing\n");
	}
}
	
void addBodyToRotationSystem(GLuint buffer, Config *config){
	// Debug
	if((config->getDebugLevel() & 0x8) == 8){
		printf("BodyRotatorSystem.cu\tAdding body to rotation system\n");
	}
	
	// Creating new cuda resource
	struct cudaGraphicsResource *resource;
	cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsRegisterFlagsNone);
	
	// Adding the new resource
	resources.push_back(resource);
}

void prepareBodyRotation(Config *config){
	for(size_t i=0; i<resources.size(); i++){
		// Making the resource available for the CUDA system
		cudaGraphicsMapResources(1, &resources[i]);
	}
}

void endBodyRotation(Config *config){
	for(size_t i=0; i<resources.size(); i++){
		// Making the resource available for OpenGl for rendering
		cudaGraphicsUnmapResources(1, &resources[i]);
	}
}

void rotateBody(int bodyNum, int numVertices, double *rotMatrix, Config *config){
	// Local vars
	double3 *bodyVertices = 0;
	size_t num_bytes_bodyVertices;
	
	// Getting the arrays
	cudaGraphicsResourceGetMappedPointer((void**)&bodyVertices, &num_bytes_bodyVertices, resources[bodyNum]);
	
	// Transferring the matrix
	cudaMemcpyToSymbol(cuda_translation_matrix, rotMatrix, 4*4*sizeof(double), 0, cudaMemcpyHostToDevice);
	
	// Executing CUDA kernel
	dim3 grid((numVertices/512) + 1);
	dim3 block(512);	
	simulateBodyRotation<<<grid, block>>>(bodyVertices, numVertices);
	
	// Error check
	cudaError_t error = cudaGetLastError();
	if(error != 0){	
		printf("BodyRotationSystem.cu\t\tError: %s\n", cudaGetErrorString(error));
	}
}

__global__ void simulateBodyRotation(double3 *vertices, int numVertices){
	// Global index
	int i = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	// Boundary check
	if(i >= numVertices)return;

	// Multiplying vertex and translation matrix, (Rotation around inclination axis and movement of whole body)
	double vx = cuda_translation_matrix[0]*vertices[i].x + cuda_translation_matrix[4]*vertices[i].y + cuda_translation_matrix[8]*vertices[i].z + cuda_translation_matrix[12];
	double vy = cuda_translation_matrix[1]*vertices[i].x + cuda_translation_matrix[5]*vertices[i].y + cuda_translation_matrix[9]*vertices[i].z + cuda_translation_matrix[13];
	double vz = cuda_translation_matrix[2]*vertices[i].x + cuda_translation_matrix[6]*vertices[i].y + cuda_translation_matrix[10]*vertices[i].z + cuda_translation_matrix[14];
	
	// Forcing sphere structure
	vertices[i].x = vx;
	vertices[i].y = vy;
	vertices[i].z = vz;
}
