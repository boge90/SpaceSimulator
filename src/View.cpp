#include "../include/View.hpp"
#include <iostream>

View::View(int height, Config *config){
	init(-1, -1, -1, height, config);
}

View::View(int height, int width, Config *config){
	init(-1, -1, width, height, config);
}

View::View(int x, int y, int width, int height, Config *config){
	init(x, y, width, height, config);
}

void View::init(int x, int y, int width, int height, Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){
		std::cout << "View.cpp\t\tInitializing(" << x << ", " << y << ", " << width << ", " << height << ", " << this << ")" << std::endl;
	}
	
	// Init
	this->red = 255;
	this->green = 255;
	this->blue = 255;
	this->x = x;
	this->y = y;
	this->width = width;
	this->height = height;
	this->clickActions = new std::vector<ViewClickedAction*>();
}

View::~View(){
	// Debug
	if((debugLevel & 0x10) == 16){			
		std::cout << "View.cpp\t\tFinalizing " << this << std::endl;
	}
	
	// Free
	delete clickActions;
}

void View::addViewClickedAction(ViewClickedAction *action){
	clickActions->push_back(action);
}

void View::draw(DrawService *service){
	service->drawLine(x, y, x+width, y, red, green, blue);
	service->drawLine(x, y, x, y+height, red, green, blue);
	service->drawLine(x, y+height, x+width, y+height, red, green, blue);
	service->drawLine(x+width, y, x+width, y+height, red, green, blue);
}

bool View::isInside(int _x, int _y){
	return _x >= x && _x <= (x+width) && _y >= y && _y <= (y+height);
}

void View::clicked(int button, int action){
	// Fire off listeners
	for(size_t i=0; i<clickActions->size(); i++){
		(*clickActions)[i]->onClick(this, button, action);
	}
}

void View::setX(int x){
	this->x = x;
}

int View::getX(void){
	return x;
}

void View::setY(int y){
	this->y = y;
}

int View::getY(void){
	return y;
}

void View::setWidth(int width){
	this->width = width;
}

int View::getWidth(void){
	return width;
}

void View::setHeight(int height){
	this->height = height;
}

int View::getHeight(void){
	return height;
}
