#include "../include/CheckBox.hpp"
#include <iostream>

CheckBox::CheckBox(std::string text, bool state): Button(text){
	// Debug
	std::cout << "CheckBox.cpp\t\tInitializing" << std::endl;
	
	// Init
	this->state = state;
	this->red = 255;
	this->green = 255;
	this->blue = 255;
	this->listeners = new std::vector<CheckBoxStateChangeAction*>();
	
	// Button listener
	Button::addViewClickedAction(this);
}

CheckBox::~CheckBox(){
	// Debug
	std::cout << "CheckBox.cpp\t\tFinalizing" << std::endl;
	
	// Free
	delete listeners;
}

bool CheckBox::getState(){
	return state;
}

void CheckBox::onClick(View *view, int button, int action){
	// Changing state
	state = !state;
	
	// Fire off listeners
	for(size_t i=0; i<listeners->size(); i++){
		(*listeners)[i]->onStateChange(this, state);
	}
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

void CheckBox::addStateChangeAction(CheckBoxStateChangeAction *action){
	listeners->push_back(action);
}
