#include "../include/BodyRotatorSystem.cuh"
#include <stdio.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// CUDA resources
static std::vector<BodyRotationUnit> resources;

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
	
void addBodyToRotationSystem(GLuint vertexBuffer, int numVertices, Config *config){
	// Debug
	if((config->getDebugLevel() & 0x8) == 8){
		printf("BodyRotatorSystem.cu\tAdding body to rotation system\n");
	}
	
	// Creating new cuda resource
	struct cudaGraphicsResource *vertexResource;
	cudaGraphicsGLRegisterBuffer(&vertexResource, vertexBuffer, cudaGraphicsRegisterFlagsNone);
	
	// Adding the new resource
	BodyRotationUnit unit;
	unit.vertexResource = vertexResource;
	unit.numVertices = numVertices;
	
	resources.push_back(unit);
}

void prepareBodyRotation(Config *config){
	for(size_t i=0; i<resources.size(); i++){
		// Making the resource available for the CUDA system
		cudaGraphicsMapResources(1, &resources[i].vertexResource);
	}
}

void endBodyRotation(Config *config){
	for(size_t i=0; i<resources.size(); i++){
		// Making the resource available for OpenGl for rendering
		cudaGraphicsUnmapResources(1, &resources[i].vertexResource);
	}
}

void rotateBody(int bodyNum, double *rotMatrix, Config *config){
	// Local vars
	double3 *bodyVertices = 0;
	size_t num_bytes_bodyVertices;
	
	// Getting the arrays
	cudaGraphicsResourceGetMappedPointer((void**)&bodyVertices, &num_bytes_bodyVertices, resources[bodyNum].vertexResource);
	
	// Transferring the matrix
	cudaMemcpyToSymbol(cuda_translation_matrix, rotMatrix, 4*4*sizeof(double), 0, cudaMemcpyHostToDevice);
	
	// Executing CUDA kernel
	dim3 grid((resources[bodyNum].numVertices/512) + 1);
	dim3 block(512);	
	simulateBodyRotation<<<grid, block>>>(bodyVertices, resources[bodyNum].numVertices);
	
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

	// Vertex
	double3 vertex = vertices[i];

	// Multiplying vertex and translation matrix
	vertex.x = cuda_translation_matrix[0]*vertex.x + cuda_translation_matrix[4]*vertex.y + cuda_translation_matrix[8]*vertex.z + cuda_translation_matrix[12];
	vertex.y = cuda_translation_matrix[1]*vertex.x + cuda_translation_matrix[5]*vertex.y + cuda_translation_matrix[9]*vertex.z + cuda_translation_matrix[13];
	vertex.z = cuda_translation_matrix[2]*vertex.x + cuda_translation_matrix[6]*vertex.y + cuda_translation_matrix[10]*vertex.z + cuda_translation_matrix[14];
	
	// Forcing sphere structure
	vertices[i] = vertex;
}
