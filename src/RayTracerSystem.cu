#include "../include/RayTracerSystem.cuh"
#include <stdio.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

// CUDA Memory
__constant__ double cuda_body_matrix[16];
__constant__ double cuda_source_matrix[16];

// CUDA resources
static std::vector<RayTracingUnit> resources; 

// CUDA function prototypes
void __global__ simulateRaysOne(double3 bc, double3 sc, int numBodyVertices, float3 *bodyVertices, float *bodyCoverage);
void __global__ simulateRaysTwo(double3 bc, int numBodyVertices, float3 *bodyVertices, float *bodyCoverage, double3 sc, int numSourceVertices, float3 *sourceVertices, float *sourceCoverage, float intensity);
void __global__ illuminate(int numBodyVertices, float *solarCoverageBuffer);
void __global__ unilluminate(int numBodyVertices, float *solarCoverageBuffer);

void addBodyToRayTracer(GLuint vertexBuffer, GLuint solarCoverageBuffer, bool isStar, Config *config){
	// Debug
	if((config->getDebugLevel() & 0x8) == 8){	
		printf("RayTracerSystem.cu\tAdding body to ray tracer system (%d, %d, %d)\n", vertexBuffer, solarCoverageBuffer, isStar);
	}
	
	// Initializing unit
	struct cudaGraphicsResource *vertexResource;
	cudaGraphicsGLRegisterBuffer(&vertexResource, vertexBuffer, cudaGraphicsRegisterFlagsNone);
	
	struct cudaGraphicsResource *solarCoverageResource;
	cudaGraphicsGLRegisterBuffer(&solarCoverageResource, solarCoverageBuffer, cudaGraphicsRegisterFlagsNone);
	
	RayTracingUnit unit;
	unit.solarCoverageBuffer = solarCoverageResource;
	unit.vertexBuffer = vertexResource;
	unit.isStar = isStar;
	
	// Adding body to body list
	resources.push_back(unit);
}

void rayTracerSimulateRaysOne(int starIndex, double x1, double y1, double z1, int bodyIndex, double x2, double y2, double z2, int numVertices, double *mat){
	// Local vars
	float3 *bodyVertices = 0;
	float *solarCoverage = 0;
	size_t num_bytes_bodyVertices;
	size_t num_bytes_solarCoverage;
	
	// Transferring the translation matrix
	cudaMemcpyToSymbol(cuda_body_matrix, mat, 4*4*sizeof(double), 0, cudaMemcpyHostToDevice);
	
	// Getting the arrays
	cudaGraphicsResourceGetMappedPointer((void**)&bodyVertices, &num_bytes_bodyVertices, resources[bodyIndex].vertexBuffer);
	cudaGraphicsResourceGetMappedPointer((void**)&solarCoverage, &num_bytes_solarCoverage, resources[bodyIndex].solarCoverageBuffer);
	
	int threads = 512;
	if(numVertices < threads){
		threads = numVertices;
	}
	
	dim3 grid((numVertices/512) + 1);
	dim3 block(threads);
	
	
	simulateRaysOne<<<grid, block>>>(make_double3(x2,y2,z2), make_double3(x1,y1,z1), numVertices, bodyVertices, solarCoverage);
}

void rayTracerSimulateRaysTwo(int sourceIndex, double x1, double y1, double z1, int numSourceVertices, int bodyIndex, double x2, double y2, double z2, int numBodyVertices, double *bodyMat, double *sourceMat, float intensity){
	// Local vars
	float3 *sourceVertices = 0;
	float3 *bodyVertices = 0;
	float *sourceCoverage = 0;
	float *bodyCoverage = 0;
	size_t num_bytes_sourceVertices;
	size_t num_bytes_bodyVertices;
	size_t num_bytes_sourceCoverage;
	size_t num_bytes_bodyCoverage;
	
	// Transferring the translation matrix
	cudaMemcpyToSymbol(cuda_body_matrix, bodyMat, 4*4*sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cuda_source_matrix, sourceMat, 4*4*sizeof(double), 0, cudaMemcpyHostToDevice);
	
	// Getting the arrays
	cudaGraphicsResourceGetMappedPointer((void**)&sourceVertices, &num_bytes_sourceVertices, resources[sourceIndex].vertexBuffer);
	cudaGraphicsResourceGetMappedPointer((void**)&sourceCoverage, &num_bytes_sourceCoverage, resources[sourceIndex].solarCoverageBuffer);
	
	cudaGraphicsResourceGetMappedPointer((void**)&bodyVertices, &num_bytes_bodyVertices, resources[bodyIndex].vertexBuffer);
	cudaGraphicsResourceGetMappedPointer((void**)&bodyCoverage, &num_bytes_bodyCoverage, resources[bodyIndex].solarCoverageBuffer);
	
	int threads = 1024;
	if(numBodyVertices < threads){
		threads = numBodyVertices;
	}
	
	dim3 grid((numBodyVertices/1024) + 1);
	dim3 block(threads);
	
	if(resources[sourceIndex].isStar){ // Star light does not need the accuracy of level TWO
		simulateRaysOne<<<grid, block>>>(make_double3(x2,y2,z2), make_double3(x1,y1,z1), numBodyVertices, bodyVertices, bodyCoverage);
	}else{	
		simulateRaysTwo<<<grid, block>>>(make_double3(x2, y2, z2), numBodyVertices, bodyVertices, bodyCoverage, make_double3(x1, y1, z1), numSourceVertices, sourceVertices, sourceCoverage, intensity);
	}
}

void rayTracerIllunimate(int index, int numBodyVertices){
	// Local vars
	float *solarCoverage = 0;
	size_t num_bytes_solarCoverage;
	
	// Getting the arrays
	cudaGraphicsResourceGetMappedPointer((void**)&solarCoverage, &num_bytes_solarCoverage, resources[index].solarCoverageBuffer);

	dim3 grid((numBodyVertices/512) + 1);
	dim3 block(512);

	illuminate<<<grid, block>>>(numBodyVertices, solarCoverage);
}

void rayTracerUnillunimate(int index, int numBodyVertices){
	// Local vars
	float *solarCoverage = 0;
	size_t num_bytes_solarCoverage;
	
	// Getting the arrays
	cudaGraphicsResourceGetMappedPointer((void**)&solarCoverage, &num_bytes_solarCoverage, resources[index].solarCoverageBuffer);

	dim3 grid((numBodyVertices/512) + 1);
	dim3 block(512);

	unilluminate<<<grid, block>>>(numBodyVertices, solarCoverage);
}

void prepareRaySimulation(void){
	for(size_t i=0; i<resources.size(); i++){
		RayTracingUnit u = resources[i];
		
		// Mapping vertex buffer
		cudaGraphicsMapResources(1, &resources[i].vertexBuffer);	
		cudaGraphicsMapResources(1, &resources[i].solarCoverageBuffer);	
	}
}

void finalizeRaySimulation(void){
	for(size_t i=0; i<resources.size(); i++){
		RayTracingUnit u = resources[i];
		
		// Mapping vertex buffer
		cudaGraphicsUnmapResources(1, &resources[i].vertexBuffer);	
		cudaGraphicsUnmapResources(1, &resources[i].solarCoverageBuffer);	
	}
}

void __global__ simulateRaysOne(double3 bc, double3 sc, int numBodyVertices, float3 *bodyVertices, float *bodyCoverage){
	// Global thread index
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// Will spawn some extra threads which must be terminated
	if(index >= numBodyVertices){return;}

	// Vertex data
	float lightIntensity = 0.f;
	float3 bodyNormal = bodyVertices[index];
	
	double vx = cuda_body_matrix[0]*bodyNormal.x + cuda_body_matrix[4]*bodyNormal.y + cuda_body_matrix[8]*bodyNormal.z + cuda_body_matrix[12];
	double vy = cuda_body_matrix[1]*bodyNormal.x + cuda_body_matrix[5]*bodyNormal.y + cuda_body_matrix[9]*bodyNormal.z + cuda_body_matrix[13];
	double vz = cuda_body_matrix[2]*bodyNormal.x + cuda_body_matrix[6]*bodyNormal.y + cuda_body_matrix[10]*bodyNormal.z + cuda_body_matrix[14];
	
	bodyNormal.x = vx-bc.x;
	bodyNormal.y = vy-bc.y;
	bodyNormal.z = vz-bc.z;
	
	double planEq = bodyNormal.x * (sc.x - vx) + bodyNormal.y * (sc.y - vy) + bodyNormal.z * (sc.z - vz);
	
	if(planEq > 0){
		lightIntensity = 1.f;
	}
	
	// Setting vertex color
	bodyCoverage[index] = lightIntensity;
}

void __global__ simulateRaysTwo(double3 bc, int numBodyVertices, float3 *bodyVertices, float *bodyCoverage, double3 sc, int numSourceVertices, float3 *sourceVertices, float *sourceCoverage, float intensity){
	// Global thread index
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// Will spawn some extra threads which must be terminated
	if(index >= numBodyVertices){return;}

	// Body intensity	
	float lightIntensity = bodyCoverage[index];
	if(lightIntensity == 1.f){return;}

	// Vector FROM source TO body
	float3 dirVec;
	dirVec.x = bc.x - sc.x;
	dirVec.y = bc.y - sc.y;
	dirVec.z = bc.z - sc.z;
	
	// Vertex data
	float3 bodyNormal = bodyVertices[index];
	double3 bodyVertex;
	
	bodyVertex.x = cuda_body_matrix[0]*bodyNormal.x + cuda_body_matrix[4]*bodyNormal.y + cuda_body_matrix[8]*bodyNormal.z + cuda_body_matrix[12];
	bodyVertex.y = cuda_body_matrix[1]*bodyNormal.x + cuda_body_matrix[5]*bodyNormal.y + cuda_body_matrix[9]*bodyNormal.z + cuda_body_matrix[13];
	bodyVertex.z = cuda_body_matrix[2]*bodyNormal.x + cuda_body_matrix[6]*bodyNormal.y + cuda_body_matrix[10]*bodyNormal.z + cuda_body_matrix[14];
	bodyNormal.x = bodyVertex.x-bc.x;
	bodyNormal.y = bodyVertex.y-bc.y;
	bodyNormal.z = bodyVertex.z-bc.z;
	
	int count = 0;
	for(int i=0; i<numSourceVertices; i++){
		// Get source coverage
		float sourceCov = sourceCoverage[i];
		if(sourceCov <= 0.2f){continue;}
		
		// Vertex and direction
		float3 sourceNormal = sourceVertices[i];
		double3 sourceVertex;
		
		// Source vertex and normal
		sourceVertex.x = cuda_source_matrix[0]*sourceNormal.x + cuda_source_matrix[4]*sourceNormal.y + cuda_source_matrix[8]*sourceNormal.z + cuda_source_matrix[12];
		sourceVertex.y = cuda_source_matrix[1]*sourceNormal.x + cuda_source_matrix[5]*sourceNormal.y + cuda_source_matrix[9]*sourceNormal.z + cuda_source_matrix[13];
		sourceVertex.z = cuda_source_matrix[2]*sourceNormal.x + cuda_source_matrix[6]*sourceNormal.y + cuda_source_matrix[10]*sourceNormal.z + cuda_source_matrix[14];
	
		sourceNormal.x = sourceVertex.x-sc.x;
		sourceNormal.y = sourceVertex.y-sc.y;
		sourceNormal.z = sourceVertex.z-sc.z;
		
		// Source normal must point toward the body (Dot product)
		double norLength = sqrt(sourceNormal.x*sourceNormal.x + sourceNormal.y*sourceNormal.y + sourceNormal.z*sourceNormal.z);
		double dirLength = sqrt(dirVec.x*dirVec.x + dirVec.y*dirVec.y + dirVec.z*dirVec.z);
		double dot = sourceNormal.x*dirVec.x + sourceNormal.y*dirVec.y + sourceNormal.z*dirVec.z;
		dot = dot/(norLength*dirLength);
		if(dot <= 0.0){continue;}
	
		// Plane Equation
		double planEq = bodyNormal.x * (sourceVertex.x - bodyVertex.x) + bodyNormal.y * (sourceVertex.y - bodyVertex.y) + bodyNormal.z * (sourceVertex.z - bodyVertex.z);
		
		if(planEq > 0){
			count++;
			lightIntensity = intensity;
		}
	}
	
	// Setting vertex color
	if(count == 0){
		bodyCoverage[index] = 0.f;
	}else{	
		bodyCoverage[index] = lightIntensity;
	}
}

void __global__ illuminate(int numBodyVertices, float *solarCoverageBuffer){
	// Global thread index
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// Will spawn some extra threads which must be terminated
	if(index >= numBodyVertices){return;}
	
	// Setting vertex color
	solarCoverageBuffer[index] = 1.f;
}

void __global__ unilluminate(int numBodyVertices, float *solarCoverageBuffer){
	// Global thread index
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// Will spawn some extra threads which must be terminated
	if(index >= numBodyVertices){return;}
	
	// Setting vertex color
	solarCoverageBuffer[index] = 0.f;
}
