#include "../include/Button.hpp"
#include <iostream>

Button::Button(std::string text, ViewClickedAction *action): TextView(text){
	// debug
	std::cout << "Button.cpp\t\tInitializing" << std::endl;
	
	// Adding click action
	addViewClickedAction(action);
}

Button::~Button(){
	std::cout << "Button.cpp\t\tFinalizing" << std::endl;
}

void Button::clicked(int button, int action){
	TextView::clicked(button, action);
}
