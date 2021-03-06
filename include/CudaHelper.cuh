#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include "../include/Config.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>

class CudaHelper{
	private:
		static int numGpus;
		static int renderingDeviceNumber;
		
		/**
		* Converts a boolean value to a string and is used when printing out GPU information
		**/
		static inline const char* boolToString(const bool b);
	public:
		/**
		* Finds the number of GPUs and creates a stream for each GPU, and sets the device input
		* parameter as the rendering device
		**/
		static bool init(Config *config);
		
		/**
		* Returns the number of GPUs
		**/
		static int getNumGpus(void);
		
		/**
		* Returns the id of the rendering GPU
		**/
		int getRenderingDeviceNumber(void);
};

#endif
