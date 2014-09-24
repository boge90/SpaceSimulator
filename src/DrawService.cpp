#include "../include/DrawService.hpp"

#include <iostream>
#include <assert.h>
#include <math.h>

//Drawing info

// Defines
#define PI 3.14159265359

DrawService::DrawService(int width, int height, unsigned char *pixels, Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){		
		std::cout << "DrawService.cpp\t\tInitializing" << std::endl;
	}
	
	// Init
	this->width = width;
	this->height = height;
	this->pixels = pixels;
	this->CHAR_WIDTH = 5;
	this->CHAR_HEIGHT = 9;
}

DrawService::~DrawService(void){
	if((debugLevel & 0x10) == 16){			
		std::cout << "DrawService.cpp\t\tFinalizing" << std::endl;
	}
}

void DrawService::fill(unsigned char r, unsigned char g, unsigned char b){
	int i;
    for(i = 0; i < width*height; i++){
    	pixels[i*3+2] = r;	
    	pixels[i*3+1] = g;
    	pixels[i*3+0] = b;
    }
}

void DrawService::fillArea(int x, int y, unsigned char r, unsigned char g, unsigned char b){
	setPixel(x, y, r, g, b);
	
	//Check neighbors (x+1, y)
	if(getRed(x+1, y) != r || getGreen(x+1, y) != g || getBlue(x+1, y) != b){
		fillArea(x+1, y, r, g, b);
	}
	
	//Check neighbors (x-1, y)
	if(getRed(x-1, y) != r || getGreen(x-1, y) != g || getBlue(x-1, y) != b){
		fillArea(x-1, y, r, g, b);
	}
	
	//Check neighbors (x, y+1)
	if(getRed(x, y+1) != r || getGreen(x, y+1) != g || getBlue(x, y+1) != b){
		fillArea(x, y+1, r, g, b);
	}
	
	//Check neighbors (x, y-1)
	if(getRed(x, y-1) != r || getGreen(x, y-1) != g || getBlue(x, y-1) != b){
		fillArea(x, y-1, r, g, b);
	}
}

void DrawService::drawChar(int x, int y, char c, unsigned char r, unsigned char g, unsigned char b, int size, bool fill){
	switch(c){
		case 'A':
		{
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x+5, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x+5, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+4, y+4, r, g, b);
			setPixel(x+5, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x+5, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x+5, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x+5, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+5, y+8, r, g, b);
			break;
		}
		case 'B':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+1, r, g, b);
			setPixel(x+3, y+2, r, g, b);
			setPixel(x+3, y+3, r, g, b);
			setPixel(x+3, y+5, r, g, b);
			setPixel(x+3, y+6, r, g, b);
			setPixel(x+3, y+7, r, g, b);
			break;
		}
		case 'C':
		{
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			setPixel(x+4, y+8, r, g, b);
			break;
		}
		case 'D':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+1, r, g, b);
			setPixel(x+3, y+7, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+4, y+4, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			break;
		}
		case 'E':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			break;
		}
		case 'F':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y, r, g, b);
			break;
		}
		case 'G':
		{
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+4, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			break;
		}
		case 'H':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+4, y+4, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+4, y+8, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			break;
		}
		case 'I':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			break;
		}
		case 'J':
		{
			setPixel(x+4, y, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+4, y+4, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+3, y+7, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			setPixel(x, y+6, r, g, b);
			break;
		}
		case 'K':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y+3, r, g, b);
			setPixel(x+2, y+5, r, g, b);
			setPixel(x+3, y+1, r, g, b);
			setPixel(x+3, y+2, r, g, b);
			setPixel(x+3, y+6, r, g, b);
			setPixel(x+3, y+7, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+4, y+8, r, g, b);
			break;
		}
		case 'L':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			setPixel(x+4, y+8, r, g, b);
			break;
		}
		case 'M':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y+1, r, g, b);
			setPixel(x+2, y+2, r, g, b);
			setPixel(x+3, y+2, r, g, b);
			setPixel(x+3, y+3, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+5, y, r, g, b);
			setPixel(x+6, y, r, g, b);
			setPixel(x+6, y+1, r, g, b);
			setPixel(x+6, y+2, r, g, b);
			setPixel(x+6, y+3, r, g, b);
			setPixel(x+6, y+4, r, g, b);
			setPixel(x+6, y+5, r, g, b);
			setPixel(x+6, y+6, r, g, b);
			setPixel(x+6, y+7, r, g, b);
			setPixel(x+6, y+8, r, g, b);
			break;
		}
		case 'N':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x+1, y+2, r, g, b);
			setPixel(x+1, y+3, r, g, b);
			setPixel(x+2, y+3, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+2, y+5, r, g, b);
			setPixel(x+3, y+5, r, g, b);
			setPixel(x+3, y+6, r, g, b);
			setPixel(x+3, y+7, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+4, y+4, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+4, y+8, r, g, b);
			break;
		}
		case 'O':
		{
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x+5, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x+5, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x+5, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x+5, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x+5, y+6, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			break;
		}
		case 'P':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			break;
		}
		case 'Q':
		{
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x+5, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x+5, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x+5, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x+5, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x+3, y+6, r, g, b);
			setPixel(x+5, y+6, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			setPixel(x+5, y+8, r, g, b);
			break;
		}
		case 'R':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+2, y+5, r, g, b);
			setPixel(x+3, y+6, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+4, y+8, r, g, b);
			break;
		}
		case 'S':
		{
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			break;
		}
		case 'T':
		{
			setPixel(x, y, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x+2, y+1, r, g, b);
			setPixel(x+2, y+2, r, g, b);
			setPixel(x+2, y+3, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+2, y+5, r, g, b);
			setPixel(x+2, y+6, r, g, b);
			setPixel(x+2, y+7, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			break;
		}
		case 'U':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			setPixel(x+4, y+8, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+5, y+6, r, g, b);
			setPixel(x+5, y+5, r, g, b);
			setPixel(x+5, y+4, r, g, b);
			setPixel(x+5, y+3, r, g, b);
			setPixel(x+5, y+2, r, g, b);
			setPixel(x+5, y+1, r, g, b);
			setPixel(x+5, y, r, g, b);
			break;
		}
		case 'V':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x+1, y+5, r, g, b);
			setPixel(x+1, y+6, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			setPixel(x+2, y+7, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+7, r, g, b);
			setPixel(x+3, y+6, r, g, b);
			setPixel(x+3, y+5, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+4, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+4, y, r, g, b);
			break;
		}
		case 'W':
		{
			setPixel(x, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+6, r, g, b);
			setPixel(x+2, y+7, r, g, b);
			setPixel(x+3, y+6, r, g, b);
			setPixel(x+3, y+5, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+5, y+8, r, g, b);
			setPixel(x+6, y+8, r, g, b);
			setPixel(x+6, y+7, r, g, b);
			setPixel(x+6, y+6, r, g, b);
			setPixel(x+6, y+5, r, g, b);
			setPixel(x+6, y+4, r, g, b);
			setPixel(x+6, y+3, r, g, b);
			setPixel(x+6, y+2, r, g, b);
			setPixel(x+6, y+1, r, g, b);
			setPixel(x+6, y, r, g, b);
			break;
		}
		case 'X':
		{
			setPixel(x, y, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x+1, y+2, r, g, b);
			setPixel(x+2, y+3, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+6, y, r, g, b);
			setPixel(x+5, y+1, r, g, b);
			setPixel(x+5, y+2, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+2, y+5, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+1, y+6, r, g, b);
			setPixel(x+5, y+6, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			setPixel(x+5, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+6, y+8, r, g, b);
			break;
		}
		case 'Y':
		{
			setPixel(x, y, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x+1, y+2, r, g, b);
			setPixel(x+2, y+3, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+6, y, r, g, b);
			setPixel(x+5, y+1, r, g, b);
			setPixel(x+5, y+2, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+3, y+5, r, g, b);
			setPixel(x+3, y+6, r, g, b);
			setPixel(x+3, y+7, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			break;
		}
		case 'Z':
		{
			setPixel(x, y, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x+1, y+2, r, g, b);
			setPixel(x+1, y+3, r, g, b);
			setPixel(x+2, y+3, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+2, y+5, r, g, b);
			setPixel(x+3, y+5, r, g, b);
			setPixel(x+3, y+6, r, g, b);
			setPixel(x+3, y+7, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			setPixel(x+4, y+8, r, g, b);
			break;
		}
		case '0':
		{
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+4, y+4, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			break;
		}
		case '1':
		{
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x+2, y+1, r, g, b);
			setPixel(x+3, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x+3, y+2, r, g, b);
			setPixel(x+3, y+3, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+3, y+5, r, g, b);
			setPixel(x+3, y+6, r, g, b);
			setPixel(x+3, y+7, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			break;
		}
		case '2':
		{
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x+2, y+1, r, g, b);
			setPixel(x+3, y+1, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+3, y+3, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+1, y+5, r, g, b);
			setPixel(x+2, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x+1, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			setPixel(x+4, y+8, r, g, b);
			break;
		}
		case '3':
		{
			setPixel(x, y+1, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			break;
		}
		case '4':
		{
			setPixel(x+3, y, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x+2, y+1, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+1, y+2, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+4, y+4, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+4, y+8, r, g, b);
			break;
		}
		case '5':
		{
			setPixel(x, y, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+3, y+5, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			break;
		}
		case '6':
		{
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x+1, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			setPixel(x+3, y+7, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+3, y+5, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+1, y+5, r, g, b);
			break;
		}
		case '7':
		{
			setPixel(x, y, r, g, b);
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x+4, y, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x+3, y+2, r, g, b);
			setPixel(x+3, y+3, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+2, y+5, r, g, b);
			setPixel(x+2, y+6, r, g, b);
			setPixel(x+2, y+7, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			break;
		}
		case '8':
		{
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x, y+5, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x, y+6, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			break;
		}
		case '9':
		{
			setPixel(x+1, y, r, g, b);
			setPixel(x+2, y, r, g, b);
			setPixel(x+3, y, r, g, b);
			setPixel(x, y+1, r, g, b);
			setPixel(x+4, y+1, r, g, b);
			setPixel(x, y+2, r, g, b);
			setPixel(x+4, y+2, r, g, b);
			setPixel(x, y+3, r, g, b);
			setPixel(x+4, y+3, r, g, b);
			setPixel(x+1, y+4, r, g, b);
			setPixel(x+2, y+4, r, g, b);
			setPixel(x+3, y+4, r, g, b);
			setPixel(x+4, y+5, r, g, b);
			setPixel(x+4, y+6, r, g, b);
			setPixel(x+4, y+7, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x+2, y+8, r, g, b);
			setPixel(x+3, y+8, r, g, b);
			break;
		}
		case '.':
		{
			setPixel(x, y+8, r, g, b);
			setPixel(x+1, y+8, r, g, b);
			setPixel(x, y+7, r, g, b);
			setPixel(x+1, y+7, r, g, b);
			break;
		}
		case ' ':
		{
			break;
		}
	}
}

int DrawService::widthOf(char c){
	switch(c){
		case 'A':
		{
			return 6;
		}
		case 'B':
		{
			return 4;
		}
		case 'C':
		{
			return 5;
		}
		case 'D':
		{
			return 5;
		}
		case 'E':
		{
			return 4;
		}
		case 'F':
		{
			return 4;
		}
		case 'G':
		{
			return 5;
		}
		case 'H':
		{
			return 5;
		}
		case 'I':
		{
			return 1;
		}
		case 'J':
		{
			return 5;
		}
		case 'K':
		{
			return 5;
		}
		case 'L':
		{
			return 5;
		}
		case 'M':
		{
			return 7;
		}
		case 'N':
		{
			return 5;
		}
		case 'O':
		{
			return 6;
		}
		case 'P':
		{
			return 5;
		}
		case 'Q':
		{
			return 6;
		}
		case 'R':
		{
			return 5;
		}
		case 'S':
		{
			return 5;
		}
		case 'T':
		{
			return 5;
		}
		case 'U':
		{
			return 6;
		}
		case 'V':
		{
			return 5;
		}
		case 'W':
		{
			return 7;
		}
		case 'X':
		{
			return 7;
		}
		case 'Y':
		{
			return 7;
		}
		case 'Z':
		{
			return 5;
		}
		case '0':
		{
			return 5;
		}
		case '1':
		{
			return 4;
		}
		case '2':
		{
			return 5;
		}
		case '3':
		{
			return 5;
		}
		case '4':
		{
			return 5;
		}
		case '5':
		{
			return 5;
		}
		case '6':
		{
			return 5;
		}
		case '7':
		{
			return 5;
		}
		case '8':
		{
			return 5;
		}
		case '9':
		{
			return 5;
		}
		case '.':
		{
			return 2;
		}
		case ' ':
		{
			return 5;
		}
	}
	
	return 5;
}

void DrawService::drawRectangle(int x, int y, int reqWidth, int reqHeight, unsigned char r, unsigned char g, unsigned char b, bool fill){
	//Draw lines
	drawLine(x, y, x+reqWidth, y, r, g, b);
	drawLine(x, y, x, y+reqHeight, r, g, b);
	drawLine(x, y+reqHeight, x+reqWidth, y+reqHeight, r, g, b);
	drawLine(x+reqWidth, y, x+reqWidth, y+reqHeight, r, g, b);
	
	//Fill
	if(fill){
		fillArea(x+1, y+1, r, g, b);
	}
}

void DrawService::drawCircle(int xc, int yc, int radius, unsigned char r, unsigned char g, unsigned char b, bool fill){
	drawCircleCenter(xc+radius/2, yc+radius/2, radius, r, g, b, fill);
}

void DrawService::drawCircleCenter(int xc, int yc, int radius, unsigned char r, unsigned char g, unsigned char b, bool fill){
	//Init
	int x, y, e;
	x = 0;
	y = radius;
	e = -radius;
	
	//Draw
	while(x <= y){
		setPixel(xc + x, yc + y, r, g, b);
		setPixel(xc + y, yc + x, r, g, b);
		setPixel(xc + y, yc - x, r, g, b);
		setPixel(xc + x, yc - y, r, g, b);
		setPixel(xc - x, yc - y, r, g, b);
		setPixel(xc - y, yc - x, r, g, b);
		setPixel(xc - y, yc + x, r, g, b);
		setPixel(xc - x, yc + y, r, g, b);
		
		e = e + 2*x + 2;
		x = x + 1;
		if(e >= 0){
			e = e - 2*y + 2;
			y = y - 1;
		}
	}
	
	//Fill
	if(fill == 1){
		fillArea(xc, yc, r, g, b);
	}
}

void DrawService::drawLine(int xs, int ys, int xe, int ye, unsigned char r, unsigned char g, unsigned char b){
	//Init
	int x, y, e, dx, dy, inc_x, inc_y;
	dx = xe - xs;
	dy = ye - ys;
	
	//Direction check
	if(xs >= xe && ys >= ye){
		//Up left
		dx = -dx;
		dy = -dy;
		inc_x = 1;
		inc_y = 1;
		int temp;
		temp = xs;
		xs = xe;
		xe = temp;
		temp = ys;
		ys = ye;
		ye = temp;
	}else if(xs <= xe && ys >= ye){
		//Up right
		dy = -dy;
		inc_x = 1;
		inc_y = -1;
	}else if(xs > xe && ys <= ye){
		//Down left
		dx = -dx;
		int temp;
		temp = xs;
		xs = xe;
		xe = temp;
		temp = ys;
		ys = ye;
		ye = temp;
		inc_x = 1;
		inc_y = -1;
	}else{
		//Down right
		inc_x = 1;
		inc_y = 1;
	}
	
	x = xs;
	y = ys;
	e = -(dx >> 1);
	
	//More than 45 degree slope
	if(dy != 0 && dx/dy < 1){
		if(dx != 0){		
			//Degree != 90
			while(x <= xe){		
				setPixel(x, y, r, g, b);
				y = y + inc_y;
				e = e + dx;
		
				if(e >= 0){
					x = x + inc_x;
					e = e - dy;
				}
			}
		}else{
			//Vertical line
			while(y <= ye){
				setPixel(x, y, r, g, b);
				y = y + inc_y;
			}
		}
	}else{
		while(x <= xe){		
			setPixel(x, y, r, g, b);
			x = x + inc_x;
			e = e + dy;
		
			if(e >= 0){
				y = y + inc_y;
				e = e - dx;
			}
		}
	}
}

void DrawService::setPixel(int x, int y, unsigned char r, unsigned char g, unsigned char b){
	//Verify coordinates in screen
	assert(x < width && x >= 0);
	assert(y < height && y >= 0);
	
	//Set pixel
	pixels[y*width*3 + x*3 + 2] = r;
	pixels[y*width*3 + x*3 + 1] = g;
	pixels[y*width*3 + x*3 + 0] = b;
}

inline unsigned char DrawService::getRed(int x, int y){
	return pixels[y*width*3 + x*3 + 2];
}

inline unsigned char DrawService::getGreen(int x, int y){
	return pixels[y*width*3 + x*3 + 1];
}

inline unsigned char DrawService::getBlue(int x, int y){
	return pixels[y*width*3 + x*3];
}
