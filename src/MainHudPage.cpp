#include "../include/MainHudPage.hpp"
#include <iostream>
#include "../include/FloatInputView.hpp"

MainHudPage::MainHudPage(int x, int y, int width, int height, Simulator *simulator): HudPage(x, y, width, height, 1){
	// Debug
	std::cout << "MainHudPage.cpp\t\tInitializing" << std::endl;
	
	// Init
	this->simulator = simulator;
	this->fpsView = new TextView("");
	this->timeView = new TextView("");
	this->cullBackfaceBox = new CheckBox("CULL BACK FACE", true);
	this->wireframeBox = new CheckBox("WIREFRAME", false);
	this->futureBodyPathBox = new CheckBox("FUTURE PATH", false);
	this->futureBodyInputView = new IntegerInputView("FUTURE BODY PATH");
	this->exitButton = new Button("EXIT");
	
	// Adding listeners
	cullBackfaceBox->addStateChangeAction(this);
	wireframeBox->addStateChangeAction(this);
	futureBodyPathBox->addStateChangeAction(this);
	exitButton->addViewClickedAction(this);
	futureBodyInputView->addIntegerInputAction(this);
	
	// Add view
	addChild(fpsView);
	addChild(timeView);
	addChild(cullBackfaceBox);
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
	}else if(box == cullBackfaceBox){
		if(newState){
			glEnable(GL_CULL_FACE);
		}else{
			glDisable(GL_CULL_FACE);
		}
	}
}

void MainHudPage::onIntegerInput(IntegerInputView *view, int value){
	simulator->getBodyTracer()->calculateFuturePath(value);
}

void MainHudPage::draw(DrawService *drawService){
	// SUper
	HudPage::draw(drawService);
	
	// Updating time
	std::string text = "TIME ";
	text += std::to_string(simulator->getTime());	
	this->timeView->setText(text);
	
	// Updating FPS
	text = "FPS ";
	text += std::to_string(simulator->getFrame()->getFPS());	
	this->fpsView->setText(text);
}
