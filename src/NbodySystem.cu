#include "../include/NbodySystem.cuh"
#include <stdio.h>
#include <vector>
#include <cuda_runtime_api.h>

// data
static size_t numBodies;
static double *cudaPositions;
static double *cudaVelocity;
static double *cudaForces;
static double *cudaMass;

// Constants
__constant__ double cudaG;
__constant__ double cudaDt;

// Kernel prototypes
void __global__ cudaCalculateForce(size_t numBodies, double *cudaPositions, double *cudaForces, double *cudaMass);
void __global__ cudaUpdatePositions(size_t numBodies, double *cudaPositions, double *cudaVelocity, double *cudaForces, double *cudaMass);

void initializeNbodySystem(double G, double dt, double *positions, double *velocity, double *mass, size_t in_numBodies, Config *config){
	if((config->getDebugLevel() & 0x10) == 16){	
		printf("NbodySystem.cu\t\tInitializing\n");
	}
	
	// Init
	numBodies = in_numBodies;
	
	// Allocating memory
	cudaMalloc(&cudaPositions, numBodies*3*sizeof(double));
	cudaMalloc(&cudaVelocity, numBodies*3*sizeof(double));
	cudaMalloc(&cudaForces, numBodies*3*sizeof(double));
	cudaMalloc(&cudaMass, numBodies*sizeof(double));
	
	// Setting initial data
	cudaMemcpy(cudaPositions, positions, numBodies*3*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaVelocity, velocity, numBodies*3*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaMass, mass, numBodies*sizeof(double), cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(cudaG, &G, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cudaDt, &dt, sizeof(double), 0, cudaMemcpyHostToDevice);
	
	// Error check
	cudaError_t error = cudaGetLastError();
	if(error != 0){	
		printf("NbodySystem.cu\t\tError: %s\n", cudaGetErrorString(error));
	}
}

void update(double *newPositions, double *newVelocities){
	// CUDA
	dim3 grid((numBodies/512) + 1);
	dim3 block(512);
	cudaCalculateForce<<<grid, block>>>(numBodies, cudaPositions, cudaForces, cudaMass);
	cudaUpdatePositions<<<grid, block>>>(numBodies, cudaPositions, cudaVelocity, cudaForces, cudaMass);
	
	// Getting new data
	cudaMemcpy(newPositions, cudaPositions, numBodies*3*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(newVelocities, cudaVelocity, numBodies*3*sizeof(double), cudaMemcpyDeviceToHost);
	
	// Error check
	cudaError_t error = cudaGetLastError();
	if(error != 0){
		printf("NbodySystem.cu\t\tError: %s\n", cudaGetErrorString(error));
	}
}

// has speedup potential by using SHARED memory
// 48 KiB can contain the data needed for 614 bodies (double3 + double3 + double3 + double)
void __global__ cudaCalculateForce(size_t numBodies, double *cudaPositions, double *cudaForces, double *cudaMass){
	size_t bodyId = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(bodyId >= numBodies){return;}
	
	// Initialize force
	double3 position;
	position.x = cudaPositions[bodyId*3 + 0];
	position.y = cudaPositions[bodyId*3 + 1];
	position.z = cudaPositions[bodyId*3 + 2];
	double3 force;
	force.x = 0.0;
	force.y = 0.0;
	force.z = 0.0;
	double mass = cudaMass[bodyId];
	
	// Looping bodies
	for(size_t i=0; i<numBodies; i++){
		if(i != bodyId){
			// Calculating distance between bodies
			double dist_x = cudaPositions[i*3 + 0] - position.x;
			double dist_y = cudaPositions[i*3 + 1] - position.y;
			double dist_z = cudaPositions[i*3 + 2] - position.z;
			
			double abs_dist = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
			abs_dist = abs_dist*abs_dist*abs_dist;
			
			// Updating force
			force.x += (cudaG * mass * cudaMass[i])/abs_dist * dist_x;
			force.y += (cudaG * mass * cudaMass[i])/abs_dist * dist_y;
			force.z += (cudaG * mass * cudaMass[i])/abs_dist * dist_z;
		}
	}
	
	cudaForces[bodyId*3 + 0] = force.x;	
	cudaForces[bodyId*3 + 1] = force.y;	
	cudaForces[bodyId*3 + 2] = force.z;
}

void __global__ cudaUpdatePositions(size_t numBodies, double *cudaPositions, double *cudaVelocity, double *cudaForces, double *cudaMass){
	int bodyId = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(bodyId >= numBodies){return;}
	
	// Reading body data
	double mass;
	mass = cudaMass[bodyId];
	
	double3 force;
	force.x = cudaForces[bodyId*3 + 0];
	force.y = cudaForces[bodyId*3 + 1];
	force.z = cudaForces[bodyId*3 + 2];
	
	double3 position;
	position.x = cudaPositions[bodyId*3 + 0];
	position.y = cudaPositions[bodyId*3 + 1];
	position.z = cudaPositions[bodyId*3 + 2];
	
	double3 velocity;
	velocity.x = cudaVelocity[bodyId*3 + 0];
	velocity.y = cudaVelocity[bodyId*3 + 1];
	velocity.z = cudaVelocity[bodyId*3 + 2];
	
	// Calculating delta
	double3 delta;
	delta.x = cudaDt * velocity.x;
	delta.y = cudaDt * velocity.y;
	delta.z = cudaDt * velocity.z;
	
	// Updating new position based on delta
	position.x += delta.x;
	position.y += delta.y;
	position.z += delta.z;
	
	cudaPositions[bodyId*3 + 0] = position.x;
	cudaPositions[bodyId*3 + 1] = position.y;
	cudaPositions[bodyId*3 + 2] = position.z;
	
	// Updating new velocity
	velocity.x += cudaDt * force.x/mass;
	velocity.y += cudaDt * force.y/mass;
	velocity.z += cudaDt * force.z/mass;
	
	cudaVelocity[bodyId*3 + 0] = velocity.x;
	cudaVelocity[bodyId*3 + 1] = velocity.y;
	cudaVelocity[bodyId*3 + 2] = velocity.z;
}
