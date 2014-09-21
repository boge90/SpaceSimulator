#include "../include/DrawService.hpp"

#include <iostream>
#include <assert.h>
#include <math.h>

//Drawing info

// Defines
#define PI 3.14159265359

DrawService::DrawService(int width, int height, unsigned char *pixels){
	//Print
	std::cout << "DrawService.cpp\t\tInitializing" << std::endl;
	
	// Init
	this->width = width;
	this->height = height;
	this->pixels = pixels;
	this->CHAR_WIDTH = 5;
	this->CHAR_HEIGHT = 9;
}

DrawService::~DrawService(void){
	std::cout << "DrawService.cpp\t\tFinalizing" << std::endl;
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
			setPixel(x+2, y, r, g, b);
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
		case ' ':
		{
			break;
		}
	}
}

void DrawService::drawRectangle(int xc, int yc, int width, int heigth, unsigned char r, unsigned char g, unsigned char b, int degree, bool fill){
	drawRectangleCenter(xc+width/2, yc+heigth/2, width, heigth, r, g, b, degree, fill);
}

void DrawService::drawRectangleCenter(int xc, int yc, int width, int heigth, unsigned char r, unsigned char g, unsigned char b, int degree, bool fill){
	//Init
	int x1, y1, x2, y2, x3, y3, x4, y4;
	int dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4;
	float radians;
	float sin_val, cos_val;
	
	//Calculate initial corners
	x1 = xc - width/2;
	y1 = yc - heigth/2;
	
	x2 = xc + width/2;
	y2 = yc - heigth/2;
	
	x3 = xc + width/2;
	y3 = yc + heigth/2;
	
	x4 = xc - width/2;
	y4 = yc + heigth/2;
	
	//Rotation
	radians = degree*PI/180.0f;
	cos_val = cosf(radians);
	sin_val = sinf(radians);
	dx1 = x1*cos_val - y1*sin_val + (-xc*cos_val + yc*sin_val + xc);
	dy1 = x1*sin_val + y1*cos_val + (-xc*sin_val - yc*cos_val + yc);
	
	dx2 = x2*cos_val - y2*sin_val + (-xc*cos_val + yc*sin_val + xc);
	dy2 = x2*sin_val + y2*cos_val + (-xc*sin_val - yc*cos_val + yc);
	
	dx3 = x3*cos_val - y3*sin_val + (-xc*cos_val + yc*sin_val + xc);
	dy3 = x3*sin_val + y3*cos_val + (-xc*sin_val - yc*cos_val + yc);
	
	dx4 = x4*cos_val - y4*sin_val + (-xc*cos_val + yc*sin_val + xc);
	dy4 = x4*sin_val + y4*cos_val + (-xc*sin_val - yc*cos_val + yc);
	
	//Draw lines
	drawLine(dx1, dy1, dx2, dy2, r, g, b);
	drawLine(dx2, dy2, dx3, dy3, r, g, b);
	drawLine(dx3, dy3, dx4, dy4, r, g, b);
	drawLine(dx4, dy4, dx1, dy1, r, g, b);
	
	//Fill
	if(fill == 1){
		fillArea(xc, yc, r, g, b);
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
