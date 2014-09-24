#ifndef BMP_SERVICE_H
#define BMP_SERVICE_H

#include "../include/BMP.hpp"
#include "../include/Config.hpp"

class BmpService{
	private:
		static long memoryUsed;
		
	public:
		static BMP* loadImage(const char *path, Config *config);
		static void freeImage(BMP *bmp, Config *config);
};

#endif
