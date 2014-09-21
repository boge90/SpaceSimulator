#include "../include/TextView.hpp"
#include <iostream>

TextView::TextView(std::string text): View(30){
	// Debug
	std::cout << "TextView.cpp\t\tInitializing" << std::endl;

	this->text = text;
}

TextView::~TextView(void){
	// Debug
	std::cout << "TextView.cpp\t\tFinalizing" << std::endl;
}

void TextView::draw(DrawService *drawService){
	// Super
	View::draw(drawService);
	
	// Drawing text
	const char *string = text.c_str();
	size_t size = text.size();
	int pos = x+10;
	
	for(size_t i=0; i<size; i++){
		drawService->drawChar(pos, y+5, string[i], 255, 255, 255, 1, false);
		pos += 10;
	}
}
