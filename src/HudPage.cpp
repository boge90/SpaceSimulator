#include "../include/HudPage.hpp"
#include <iostream>

HudPage::HudPage(int x, int y, int width, int height, int number): ListLayout(x, y, width, height){
	// Debug
	std::cout << "HudPage.cpp\t\tInitializing" << std::endl;
	
	// Init
	std::string text = "PAGE ";
	text.append(std::to_string(number));
	
	this->number = number;
	this->numberView = new TextView(text);
	
	// Super
	addChild(numberView);
}

HudPage::~HudPage(void){
	std::cout << "HudPage.cpp\t\tFinalizing" << std::endl;
}
