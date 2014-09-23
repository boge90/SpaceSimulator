#include "../include/TextView.hpp"
#include <iostream>

TextView::TextView(std::string text): View(30){
	// Debug
	std::cout << "TextView.cpp\t\tInitializing" << std::endl;

	this->text = text;
	this->leftPadding = 10;
	this->topPadding = 5;
	this->charPadding = 2;
}

TextView::~TextView(void){
	// Debug
	std::cout << "TextView.cpp\t\tFinalizing" << std::endl;
}

void TextView::draw(DrawService *drawService){
	// Super
	View::draw(drawService);
	
	// Remove previous text
	if(previousText.size() > 0){
		drawText(previousText, drawService, 0, 0, 0);
		previousText = "";
	}
	
	// Draw current text
	drawText(text, drawService, 255, 255, 255);
}

void TextView::setText(std::string text){
	this->previousText = this->text;
	this->text = text;
}

std::string* TextView::getText(void){
	return &text;
}

void TextView::drawText(std::string text, DrawService *drawService, unsigned char r, unsigned char g, unsigned char b){
	const char *string = text.c_str();
	size_t size = text.size();
	int pos = x+leftPadding;
	
	for(size_t i=0; i<size; i++){
		drawService->drawChar(pos, y+topPadding, string[i], r, g, b, 1, false);
		pos += drawService->widthOf(string[i]) + charPadding;
	}
}
