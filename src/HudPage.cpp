#include "../include/HudPage.hpp"
#include <iostream>

HudPage::HudPage(int x, int y, int width, int height, int number): ListLayout(x, y, width, height){
	// Debug
	std::cout << "HudPage.cpp\t\tInitializing" << std::endl;
	
	// Init
	this->number = number;
	this->numberView = new TextView("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
	
	// Super
	addChild(numberView);
}

HudPage::~HudPage(void){
	std::cout << "HudPage.cpp\t\tFinalizing" << std::endl;
}

void HudPage::draw(DrawService *service){
	ListLayout::draw(service);
}
