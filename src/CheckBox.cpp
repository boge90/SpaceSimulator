#include "../include/CheckBox.hpp"
#include <iostream>

CheckBox::CheckBox(std::string text): Button(text, this){
	// Debug
	std::cout << "CheckBox.cpp\t\tInitializing" << std::endl;
	
	// Init
	this->state = false;
	this->red = 255;
	this->green = 255;
	this->blue = 255;
}

CheckBox::~CheckBox(){
	// Debug
	std::cout << "CheckBox.cpp\t\tFinalizing" << std::endl;
}

bool CheckBox::getState(){
	return state;
}

void CheckBox::viewClicked(View *view, int button, int action){
	state = !state;
}

void CheckBox::draw(DrawService *drawService){
	// Super
	Button::draw(drawService);
	
	int _x = x+leftPadding;
	size_t chars = text.size();
	const char *string = text.c_str();
	
	for(size_t i=0; i<chars; i++){
		_x += drawService->widthOf(string[i]) + charPadding;
	}
	
	drawService->drawRectangle(_x + leftPadding, y+topPadding, 9, 9, red, green, blue, false);
	
	if(state){
		drawService->drawLine(_x + leftPadding, y+topPadding, _x + leftPadding + 9, y+topPadding + 9, red, green, blue);
		drawService->drawLine(_x + leftPadding+9, y+topPadding, _x + leftPadding, y+topPadding + 9, red, green, blue);
	}else{
		drawService->drawLine(_x + leftPadding, y+topPadding, _x + leftPadding + 9, y+topPadding + 9, 0, 0, 0);
		drawService->drawLine(_x + leftPadding+9, y+topPadding, _x + leftPadding, y+topPadding + 9, 0, 0, 0);
	}
}
