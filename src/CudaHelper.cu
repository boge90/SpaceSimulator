#include "../include/CudaHelper.cuh"
#include <iostream>

int CudaHelper::numGpus = -1;
int CudaHelper::renderingDeviceNumber = -1;

inline const char* CudaHelper::boolToString(const bool b){
    return b ? "Yes" : "No";
}

bool CudaHelper::init(Config *config){
	// Debug
	if((config->getDebugLevel() & 0x10) == 16){	
		std::cout << "CudaHelper.cu\t\tInitializing GPU(s)\n";
	}
	
	// Initialize stream vector
	cudaError_t error;
	
	// Getting device count
	error = cudaGetDeviceCount(&numGpus);
	if(error != 0){ // Print out error	
		std::cout << "CudaHelper.cu\t\tcudaGetDeviceCount() gave " << cudaGetErrorString(error) << "\n";
		return false;
	}
	
	// Setting OpenGL device
	cudaGLSetGLDevice(config->getRenderingDeviceNumber());
	
	if((config->getDebugLevel() & 0x8) == 8){	
		cudaDeviceProp deviceProperties;
		for(int i=0; i<numGpus; i++){
			error = cudaGetDeviceProperties(&deviceProperties, i);
		
			if(error != 0){ // Print out error
				std::cout << "CudaHelper.cu\t\tcudaGetDeviceProperties() gave " << cudaGetErrorString(error) << "\n";
				return false;
			}
		
			std::cout << "CudaHelper.cu\t\tPrinting out device number " << i << "\n";
			std::cout << "CudaHelper.cu\t\t\tName\t\t\t= " << deviceProperties.name << "\n";
			std::cout << "CudaHelper.cu\t\t\tGlobal memory\t\t= " << deviceProperties.totalGlobalMem/(1024*1024) << " MiB\n";
			std::cout << "CudaHelper.cu\t\t\tL2 cache\t\t= " << deviceProperties.l2CacheSize/(1024) << " KiB\n";
			std::cout << "CudaHelper.cu\t\t\tShared memory per block\t= " << deviceProperties.sharedMemPerBlock/(1024) << " KiB\n";
			std::cout << "CudaHelper.cu\t\t\tRegisters per block\t= " << deviceProperties.regsPerBlock << "\n";
			std::cout << "CudaHelper.cu\t\t\tClock rate\t\t= " << deviceProperties.clockRate << " MHz\n";
			std::cout << "CudaHelper.cu\t\t\tMemory clock rate\t= " << deviceProperties.memoryClockRate << " MHz\n";
			std::cout << "CudaHelper.cu\t\t\tMemory bus width\t= " << deviceProperties.memoryBusWidth << " bits\n";
			std::cout << "CudaHelper.cu\t\t\tMax threads per SM\t= " << deviceProperties.maxThreadsPerMultiProcessor << "\n";
			std::cout << "CudaHelper.cu\t\t\tConcurrent kernels\t= " << boolToString(deviceProperties.concurrentKernels) << "\n";
			std::cout << "CudaHelper.cu\t\t\tIntegrated\t\t= " << boolToString(deviceProperties.integrated) << "\n";
			std::cout << "CudaHelper.cu\t\t\tMax grid size\t\t= " << deviceProperties.maxGridSize[0] << ", " << deviceProperties.maxGridSize[1] << ", " << deviceProperties.maxGridSize[2] << "\n";
			std::cout << "CudaHelper.cu\t\t\tMax block size\t\t= " << deviceProperties.maxThreadsDim[0] << ", " << deviceProperties.maxThreadsDim[1] << ", " << deviceProperties.maxThreadsDim[2] << "\n";
		}
	}
	
	error = cudaGetLastError();
	if(error != 0){	
		std::cout << "CudaHelper.cu\t\tLast cuda error is " << error << "\n";
	}
	
	return true;
}

int CudaHelper::getNumGpus(void){
	return numGpus;
}

int CudaHelper::getRenderingDeviceNumber(void){
	return renderingDeviceNumber;
}
