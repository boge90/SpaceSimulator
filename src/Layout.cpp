#include "../include/Layout.hpp"
#include <iostream>

Layout::Layout(int x, int y, int width, int height, Config *config): View(x, y, width, height, config){
	// Debug
	if((debugLevel & 0x10) == 16){		
		std::cout << "Layout.cpp\t\tInitializing" << std::endl;
	}

	// Init
	children = new std::vector<View*>();
}

Layout::~Layout(){
	// Debug
	if((debugLevel & 0x10) == 16){		
		std::cout << "Layout.cpp\t\tFinalizing" << std::endl;
	}
	
	// Deleting children
	for(size_t i=0; i<children->size(); i++){
		delete (*children)[i];
	}
	
	delete children;
}

void Layout::addChild(View *view){
	children->push_back(view);
}

void Layout::draw(DrawService *drawService){
	// Super
	View::draw(drawService);

	// Calling parent
	View::draw(drawService);
	
	// Drawing children
	size_t numChildren = children->size();
	for(size_t i=0; i<numChildren; i++){
		(*children)[i]->draw(drawService);
	}
}

std::vector<View*>* Layout::getChildren(void){
	return children;
}
