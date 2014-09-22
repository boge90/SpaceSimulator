#include "../include/MainHudPage.hpp"
#include <iostream>

MainHudPage::MainHudPage(int x, int y, int width, int height, Simulator *simulator): HudPage(x, y, width, height, 1){
	// Debug
	std::cout << "MainHudPage.cpp\t\tInitializing" << std::endl;
	
	// Init
	this->simulator = simulator;
	this->wireframeBox = new CheckBox("WIREFRAME");
	this->futureBodyPathBox = new CheckBox("FUTURE PATH");
	this->futureBodyInputView = new IntegerInputView("FUTURE BODY PATH");
	this->exitButton = new Button("EXIT");
	
	// Adding listeners
	wireframeBox->addStateChangeAction(this);
	futureBodyPathBox->addStateChangeAction(this);
	exitButton->addViewClickedAction(this);
	futureBodyInputView->addIntegerInputAction(this);
	
	// Add view
	addChild(wireframeBox);
	addChild(futureBodyPathBox);
	addChild(futureBodyInputView);
	addChild(exitButton);
}

MainHudPage::~MainHudPage(){
	// Debug
	std::cout << "MainHudPage.cpp\t\tFinalizing" << std::endl;
}

void MainHudPage::onClick(View *view, int button, int action){
	if(view == exitButton){ // EXIT application
		glfwSetWindowShouldClose(simulator->getFrame()->getWindow(), GL_TRUE);
	}
}

void MainHudPage::onStateChange(CheckBox *box, bool newState){
	if(box == wireframeBox){ // Toggle wireframe
		std::vector<Body*> *bodies = simulator->getBodies();
		int size = bodies->size();
	
		for(int i=0; i<size; i++){
			(*bodies)[i]->setWireframeMode(newState);
		}
	}else if(box == futureBodyPathBox){ // Toggle body path visualization
		simulator->getBodyTracer()->setActive(newState);
	}
}

void MainHudPage::onIntegerInput(IntegerInputView *view, int value){
	simulator->getBodyTracer()->calculateFuturePath(value);
}
