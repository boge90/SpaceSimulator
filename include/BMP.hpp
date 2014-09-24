#ifndef BMP_H
#define BMP_H

#include "../include/Config.hpp"

class BMP{
	private:	
		int width;
		int height;
		unsigned char *data;
		
		// Misc
		size_t debugLevel;
	public:
		BMP(int width, int height, unsigned char *data, Config *config);
		~BMP(void);
		
		/**
		* returns a pointer to the data of the image
		**/
		unsigned char* getData(void);
		
		/**
		* returns the height of the image
		**/
		int getHeight(void);
		
		/**
		* returns the width of the image
		**/
		int getWidth(void);
};

#endif
