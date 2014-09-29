#include "../include/CameraHudPage.hpp"
#include "../include/AbstractCamera.hpp"
#include <iostream>

CameraHudPage::CameraHudPage(int x, int y, int width, int height, int number, Simulator *simulator, Config *config): HudPage(x, y, width, height, number, config){
	// Debug
	if((debugLevel & 0x10) == 16){
		std::cout << "CameraHudPage.cpp\tInitializing" << std::endl;
	}
	
	// Init
	this->initialized = false;
	this->simulator = simulator;
	this->fovView = new FloatInputView("FIELD OF VIEW ", config);
	
	// Listeners
	fovView->addFloatInputAction(this);
	
	// This
	addChild(fovView);
}

CameraHudPage::~CameraHudPage(void){
	if((debugLevel & 0x10) == 16){
		std::cout << "CameraHudPage.cpp\tFinalizing" << std::endl;
	}
}

void CameraHudPage::draw(DrawService *drawService){
	if(!initialized){	
		float fov = simulator->getFrame()->getMenu()->getActivatedCamera()->getFieldOfView();
		std::string text = "";
		text.append(std::to_string(fov));
		fovView->setInput(text);
		initialized = true;
	}
	
	// Super
	HudPage::draw(drawService);
}

void CameraHudPage::onFloatInput(FloatInputView *view, double value){
	simulator->getFrame()->getMenu()->getActivatedCamera()->setFieldOfView(value);
}
