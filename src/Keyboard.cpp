#include "../include/Keyboard.hpp"
#include "../include/KeyboardInput.hpp"

Keyboard::Keyboard(std::vector<Body*> *bodies, Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){
		std::cout << "Keyboard.cpp\t\tInitializing\n";
	}
	
	// Init
	this->bodies = bodies;
	this->activeBody = 0;
	
	// Adding listener
	KeyboardInput::getInstance()->addInputAction(this);
}

Keyboard::~Keyboard(){
	if((debugLevel & 0x10) == 16){
		std::cout << "Keyboard.cpp\t\tInitializing\n";
	}
	
	// Removing listener
	KeyboardInput::getInstance()->removeInputAction(this);
}

void Keyboard::onKeyInput(int key){
	if(key == 78){ // NEXT
		nextBody();
	}else if(key == 80){ // PREVIOUS
		previousBody();
	}
}

void Keyboard::nextBody(void){
	if(activeBody+1 <= bodies->size()-1){
		activeBody++;
		
		std::cout << *((*bodies)[activeBody]->getName()) << std::endl;
	}
}

void Keyboard::previousBody(void){
	if(activeBody >= 1){
		activeBody--;
	}
}
