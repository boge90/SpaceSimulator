#include "../include/Button.hpp"
#include <iostream>

Button::Button(std::string text, Config *config): TextView(text, config){
	// debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "Button.cpp\t\tInitializing" << std::endl;
	}
}

Button::~Button(){
	if((debugLevel & 0x10) == 16){
		std::cout << "Button.cpp\t\tFinalizing" << std::endl;
	}
}

void Button::clicked(int button, int action){
	TextView::clicked(button, action);
}
