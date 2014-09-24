#include "../include/HudPage.hpp"
#include <iostream>

HudPage::HudPage(int x, int y, int width, int height, int number, Config *config): ListLayout(x, y, width, height, config){
	// Debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "HudPage.cpp\t\tInitializing" << std::endl;
	}
	
	// Init
	std::string text = "PAGE ";
	text.append(std::to_string(number));
	
	this->number = number;
	this->numberView = new TextView(text, config);
	
	// Super
	addChild(numberView);
}

HudPage::~HudPage(void){
	if((debugLevel & 0x10) == 16){
		std::cout << "HudPage.cpp\t\tFinalizing" << std::endl;
	}
}
