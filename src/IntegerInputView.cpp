#include "../include/IntegerInputView.hpp"
#include <iostream>

IntegerInputView::IntegerInputView(std::string text): InputView(text){
	// Debug
	std::cout << "IntegerInputView.cpp\tInitializing" << std::endl;
	
	// Init
	this->listeners = new std::vector<IntegerInputAction*>();
	
	// Input listener
	InputView::addInputViewAction(this);
}

IntegerInputView::~IntegerInputView(){
	// Debug
	std::cout << "IntegerInputView.cpp\tFinalizing" << std::endl;
}

void IntegerInputView::onInput(InputView *view, std::string *input){
	// Debug
	int value = atoi(input->c_str());
	
	// IFF the user specifies a invalid number, atoi returns '0' which is then
	// added as the input value.
	InputView::setInput(std::to_string(value));
	
	// Fire of listeners
	for(size_t i=0; i<listeners->size(); i++){
		(*listeners)[i]->onIntegerInput(this, value);
	}
}

void IntegerInputView::addIntegerInputAction(IntegerInputAction *listener){
	this->listeners->push_back(listener);
}
