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
	
	// Deleting children
	for(size_t i=0; i<children->size(); i++){
		delete (*children)[i];
	}
	
	delete children;
}

void ListLayout::addChild(View *view){
	children->push_back(view);
	
	int padding = 10;
	size_t childNumber = children->size();
	
	// Setting the position of the view
	view->setX(x + padding);
	
	// Find Y coordinate
	int childY = y + padding; // Top padding
	for(size_t i=0; i<childNumber-1; i++){
		childY += (*children)[i]->getHeight() + padding; // 10 is padding
	}
	
	view->setY(childY);
	
	if(view->getWidth() < 0){
		view->setWidth(width - 2*padding);
	}
	
	// Debug
	std::cout << "Added child x = " << view->getX() << ", y = " << view->getY() << ", width = " << view->getWidth() << ", height = " << view->getHeight() << std::endl;
}

void ListLayout::draw(DrawService *drawService){
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
