#include "../include/FloatInputView.hpp"
#include <iostream>

FloatInputView::FloatInputView(std::string text, Config *config): InputView(text, config){
	// Debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "FloatInputView.cpp\tInitializing" << std::endl;
	}
	
	// Init
	this->listeners = new std::vector<FloatInputAction*>();
	
	// Input listener
	InputView::addInputViewAction(this);
}

FloatInputView::~FloatInputView(){
	// Debug
	if((debugLevel & 0x10) == 16){	
		std::cout << "FloatInputView.cpp\tFinalizing" << std::endl;
	}
	
	delete listeners;
}

void FloatInputView::onInput(InputView *view, std::string *input){
	// Parsing input value
	double value = atof(input->c_str());
	
	// IFF the user specifies a invalid number, atoi returns '0' which is then
	// added as the input value.
	InputView::setInput(std::to_string(value));
	
	// Fire of listeners
	for(size_t i=0; i<listeners->size(); i++){
		(*listeners)[i]->onFloatInput(this, value);
	}
}

void FloatInputView::addFloatInputAction(FloatInputAction *listener){
	this->listeners->push_back(listener);
}
