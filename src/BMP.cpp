#include "../include/BMP.hpp"
#include "iostream"

BMP::BMP(int width, int height, unsigned char *data){
	std::cout << "BMP.cpp\t\t\tInitializing\n";	
	
	this->width = width;
	this->height = height;
	this->data = data;
}

BMP::~BMP(void){
	std::cout << "BMP.cpp\t\t\tFinalizing\n";	
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
