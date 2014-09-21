#include "../include/NbodySystem.cuh"
#include <stdio.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// Vector of vertes resources to all bodies
//static std::vector<struct cudaGraphicsResource*> *resources = new std::vector<struct cudaGraphicsResource*>();
static std::vector<struct cudaGraphicsResource*> resources;

// Kernel prototypes
void __global__ moveBodyKernel(float3 *vertices, int num_vertices, double cx, double cy, double cz, double radius);

// CUDA Memory
__constant__ double cuda_translation_matrix[16];

void initializeNbodySystem(void){
	printf("NbodySystem.cu\t\tInitializing\n");	
}

void addBodyVertexBuffer(GLuint buffer){
	printf("NbodySystem.cu\t\tAdding vertex buffer %d\n", buffer);
	
	// Creating new cuda resource
	struct cudaGraphicsResource *resource;
	cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsRegisterFlagsNone);
	
	// Adding the new resource
	resources.push_back(resource);
}

void moveBody(int bodyIndex, int numVertices, double *translation, double cx, double cy, double cz, double radius){
	// Local vars
	float3 *vertices = 0;
	size_t num_bytes_vertices;
	
	// Getting the vertex array pointer
	cudaGraphicsMapResources(1, &resources[bodyIndex]);
	cudaGraphicsResourceGetMappedPointer((void**)&vertices, &num_bytes_vertices, resources[bodyIndex]);
	
	// Transferring the translation matrix
	cudaMemcpyToSymbol(cuda_translation_matrix, translation, 4*4*sizeof(double), 0, cudaMemcpyHostToDevice);
	
	// CUDA call
	dim3 block((numVertices/512) + 1);
	dim3 grid(512);
	moveBodyKernel<<<block, grid>>>(vertices, numVertices, cx, cy, cz, radius);
	
	// Unmapping, making ready for rendering
	cudaGraphicsUnmapResources(1, &resources[bodyIndex]);
	
	// Error check
	cudaError_t error = cudaGetLastError();
	if(error != 0){	
		printf("NbodySystem.cu\t\tError: %s\n", cudaGetErrorString(error));
	}
}

void __global__ moveBodyKernel(float3 *vertices, int num_vertices, double cx, double cy, double cz, double radius){
	// Global index
	int i = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	// Boundary check
	if(i >= num_vertices)return;

	// Multiplying vertex and translation matrix, (Rotation around inclination axis and movement of whole body)
	double vx = cuda_translation_matrix[0]*vertices[i].x + cuda_translation_matrix[1]*vertices[i].y + cuda_translation_matrix[2]*vertices[i].z + cuda_translation_matrix[3];
	double vy = cuda_translation_matrix[4]*vertices[i].x + cuda_translation_matrix[5]*vertices[i].y + cuda_translation_matrix[6]*vertices[i].z + cuda_translation_matrix[7];
	double vz = cuda_translation_matrix[8]*vertices[i].x + cuda_translation_matrix[9]*vertices[i].y + cuda_translation_matrix[10]*vertices[i].z + cuda_translation_matrix[11];
	
	// Forcing sphere structure
	vertices[i].x = vx;
	vertices[i].y = vy;
	vertices[i].z = vz;
}
