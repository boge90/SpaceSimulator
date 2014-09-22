#include "../include/MainHudPage.hpp"
#include <iostream>

#include "../include/CheckBox.hpp"

MainHudPage::MainHudPage(int x, int y, int width, int height, Simulator *simulator): HudPage(x, y, width, height, 1){
	// Debug
	std::cout << "MainHudPage.cpp\t\tInitializing" << std::endl;
	
	// Init
	this->simulator = simulator;
	this->wireframeButton = new Button("WIREFRAME", this);
	this->futureBodyPathButton = new Button("FUTURE PATH", this);
	this->exitButton = new Button("EXIT", this);
	
	// Add view
	addChild(wireframeButton);
	addChild(futureBodyPathButton);
	addChild(exitButton);
}

MainHudPage::~MainHudPage(){
	// Debug
	std::cout << "MainHudPage.cpp\t\tFinalizing" << std::endl;
}

void MainHudPage::viewClicked(View *view, int button, int action){
	if(view == wireframeButton){ // Toggle wireframe
		std::vector<Body*> *bodies = simulator->getBodies();
		int size = bodies->size();
	
		for(int i=0; i<size; i++){
			(*bodies)[i]->toogleWireFrame();
		}
	}else if(view == futureBodyPathButton){ // Toggle body path visualization
		simulator->getBodyTracer()->toggle();
	}else if(view == exitButton){ // EXIT application
		glfwSetWindowShouldClose(simulator->getFrame()->getWindow(), GL_TRUE);
	}
}
