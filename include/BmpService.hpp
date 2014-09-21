#ifndef BMP_SERVICE_H
#define BMP_SERVICE_H

#include "../include/BMP.hpp"

class BmpService{
	private:
		static long memoryUsed;
		
	public:
		static BMP* loadImage(const char *path);
		static void freeImage(BMP *bmp);
};

#endif
