#include "../include/KeyboardInput.hpp"
#include <iostream>

// Setting instance initial value
KeyboardInput* KeyboardInput::instance = NULL;

KeyboardInput::KeyboardInput(void){
	// Init
	this->listeners = new std::vector<KeyboardInputAction*>();
}

KeyboardInput::~KeyboardInput(void){
	delete listeners;
}

void KeyboardInput::addInput(int key){
	// Fireing off listenerss
	for(size_t i=0; i<listeners->size(); i++){
		(*listeners)[i]->onKeyInput(key);
	}
}


void KeyboardInput::addInputAction(KeyboardInputAction *action){
	if(getIndex(action) < 0){	
		std::cout << "KeyboardInput.cpp\tAdding KeyboarInputAction " << action << std::endl;
		listeners->push_back(action);
	}
}
	
void KeyboardInput::removeInputAction(KeyboardInputAction *action){
	std::cout << "KeyboardInput.cpp\tRemoving KeyboarInputAction " << action << std::endl;
	listeners->erase(listeners->begin() + getIndex(action));
}

int KeyboardInput::getIndex(KeyboardInputAction *action){
	for(size_t i=0; i<listeners->size(); i++){
		if((*listeners)[i] == action){
			return i;
		}
	}
	return -1;
}

KeyboardInput* KeyboardInput::getInstance(void){
	if(instance == NULL){
		instance = new KeyboardInput();
	}
	
	return instance;
}
