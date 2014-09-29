#include "../include/BmpService.hpp"
#include <iostream>
#include <stdlib.h>

long BmpService::memoryUsed = 0l;

BMP* BmpService::loadImage(const char *path, Config *config){
	// Debug
	if((config->getDebugLevel() & 0x8) == 8){	
		std::cout << "BmpService.cpp\t\tLoading image " << path << "\n";
	}
	
	// Open file
	FILE* fp = fopen(path, "rb");

	// Vars
    int width, height, offset, readNum;

	// Reading vars
    fseek(fp, 18, SEEK_SET);
    readNum = fread(&width, 4, 1, fp);
    if(readNum != 1){
    	std::cout << "BmpService.cpp\t\tThere occured an error while reading image width\n";
    }
    
    fseek(fp, 22, SEEK_SET);
    readNum = fread(&height, 4, 1, fp);
    if(readNum != 1){
    	std::cout << "BmpService.cpp\t\tThere occured an error while reading image height\n";
    }
    
    fseek(fp, 10, SEEK_SET);
    readNum = fread(&offset, 4, 1, fp);
    if(readNum != 1){
    	std::cout << "BmpService.cpp\t\tThere occured an error while reading image offset\n";
    }

	// Allocating data for image
    unsigned char *data = (unsigned char*) malloc(sizeof(unsigned char)*height*width*3);

	// Total memory usage
	memoryUsed += sizeof(unsigned char)*height*width*3;
	if((config->getDebugLevel() & 0x40) == 64){	
		std::cout << "BmpService.cpp\t\tTotal memory usage is " << memoryUsed/(1024.0*1024.0) << " MiB\n";
	}

	// Reading image
    fseek(fp, offset, SEEK_SET);
    readNum = fread(data, sizeof(unsigned char), height*width*3, fp);
    if(readNum != height*width*3){
    	std::cout << "BmpService.cpp\t\tThere occured an error while reading image data\n";
    }

	// Closing file
    fclose(fp);
	
	// Return image
	return new BMP(width, height, data, config);
}

void BmpService::freeImage(BMP *bmp, Config *config){
	memoryUsed -= sizeof(unsigned char)*bmp->getHeight()*bmp->getWidth()*3;
	free(bmp->getData());

	delete bmp;
	
	// Total memory usage
	if((config->getDebugLevel() & 0x40) == 64){	
		std::cout << "BmpService.cpp\t\tTotal memory usage is " << memoryUsed/(1024.0*1024.0) << " MiB\n";
	}
}
