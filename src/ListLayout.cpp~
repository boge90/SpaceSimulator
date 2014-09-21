#include "../include/ListLayout.hpp"
#include <iostream>

ListLayout::ListLayout(int x, int y, int width, int height): View(x, y, width, height){
	// Debug
	std::cout << "ListLayout.cpp\t\tInitializing" << std::endl;
	
	// Init
	children = new std::vector<View*>();
}

ListLayout::~ListLayout(){
	// Debug
	std::cout << "ListLayout.cpp\t\tFinalizing" << std::endl;
}

void ListLayout::addChild(View *view){
	children->push_back(view);
}

void ListLayout::draw(DrawService *drawService){
	// Calling parent
	View::draw(drawService);
	
	// Drawing children
	size_t numChildren = children->size();
	for(size_t i=0; i<numChildren; i++){
		(*children)[i]->draw(drawService);
	}
}
