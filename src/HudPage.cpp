#include "../include/HudPage.hpp"
#include <iostream>

HudPage::HudPage(int x, int y, int width, int height, std::string title, Config *config): ListLayout(x, y, width, height, config){
	// Debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "HudPage.cpp\t\tInitializing" << std::endl;
	}
	
	this->titleView = new TextView(title, config);
	
	// Super
	addChild(titleView);
}

HudPage::~HudPage(void){
	if((debugLevel & 0x10) == 16){
		std::cout << "HudPage.cpp\t\tFinalizing" << std::endl;
	}
}
