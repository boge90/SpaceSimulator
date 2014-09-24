#include "../include/BMP.hpp"
#include "iostream"

BMP::BMP(int width, int height, unsigned char *data, Config *config){
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){		
		std::cout << "BMP.cpp\t\t\tInitializing BMP image (" << width << " x " << height << ")" << std::endl;	
	}
	
	this->width = width;
	this->height = height;
	this->data = data;
}

BMP::~BMP(void){
	if((debugLevel & 0x10) == 16){	
		std::cout << "BMP.cpp\t\t\tFinalizing\n";	
	}
}

unsigned char* BMP::getData(void){
	return data;
}

int BMP::getHeight(void){
	return height;
}
		
int BMP::getWidth(void){
	return width;
}
