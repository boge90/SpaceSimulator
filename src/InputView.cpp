#include "../include/InputView.hpp"
#include "../include/KeyboardInput.hpp"

#include <GLFW/glfw3.h>
#include <iostream>

InputView::InputView(std::string text, Config *config): Button(text, config){
	// Debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "InputView.cpp\t\tInitializing" << std::endl;
	}
	
	// Init
	this->repaint = false;
	this->input = "";
	this->listeners = new std::vector<InputViewAction*>();
	
	// Adding super button listener
	Button::addViewClickedAction(this);
}

InputView::~InputView(){
	if((debugLevel & 0x10) == 16){	
		std::cout << "InputView.cpp\t\tFinalizing" << std::endl;
	}
	
	delete listeners;
}

void InputView::onClick(View *view, int button, int action){
	KeyboardInput::getInstance()->addInputAction(this);
}

void InputView::onKeyInput(int key){
	// Appending key
	if(key == GLFW_KEY_BACKSPACE){
		// Remove last
		input = input.substr(0, input.size()-1);
		repaint = true;
	}else if(key < 100){
		input += key;
	}
	
	if(key == GLFW_KEY_ENTER){
		// remove listener
		KeyboardInput::getInstance()->removeInputAction(this);
		
		// Fire of listeners
		for(size_t i=0; i<listeners->size(); i++){
			(*listeners)[i]->onInput(this, &input);
		}
	}
}

void InputView::draw(DrawService *drawService){
	//Super
	Button::draw(drawService);
	
	// Repaint if characters has been removed
	if(repaint){
		repaint = false;
		drawService->fillArea(x+1, y+1, 255, 255, 255);
		drawService->fillArea(x+1, y+1, 0, 0, 0);
	}
	
	// Finding starting position for input text
	int _x = x + leftPadding;
	_x += drawService->widthOf(text) + charPadding;

	
	_x += leftPadding;
	
	// Drawing input
	const char *string = input.c_str();
	for(size_t i=0; i<input.size(); i++){
		drawService->drawChar(_x, y+topPadding, string[i], 255, 255, 255, 1, false);
		_x += drawService->widthOf(string[i]) + charPadding;
	}
}

void InputView::addInputViewAction(InputViewAction *listener){
	listeners->push_back(listener);
}

void InputView::setInput(std::string input){
	this->input = input;
	repaint = true;
}
