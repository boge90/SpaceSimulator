#include "../include/MainHudPage.hpp"
#include "../include/DateService.hpp"

#include <iostream>

MainHudPage::MainHudPage(int x, int y, int width, int height, Simulator *simulator, Config *config): HudPage(x, y, width, height, "MAIN", config){
	// Debug
	if((debugLevel & 0x10) == 16){		
		std::cout << "MainHudPage.cpp\t\tInitializing" << std::endl;
	}
	
	// Init
	this->simulator = simulator;
	this->fpsView = new TextView("", config);
	this->timeView = new TextView("", config);
	this->pausedBox = new CheckBox("PAUSE", false, config);
	this->cullBackfaceBox = new CheckBox("CULL BACK FACE", true, config);
	this->wireframeBox = new CheckBox("WIREFRAME", false, config);
	this->fakeSizeBox = new CheckBox("FAKE SIZE", false, config);
	this->starDimmerBox = new CheckBox("STAR DIMMING", true, config);
	this->futureBodyPathBox = new CheckBox("FUTURE PATH", false, config);
	this->futureBodyInputView = new IntegerInputView("FUTURE BODY PATH", config);
	this->bodyLocatorBox = new CheckBox("BODY LOCATOR", false, config);
	this->bodyLocatorInputView = new IntegerInputView("BODY LOCATOR NUMBER", config);
	this->rayTracingLevelView = new SelectView<int>("RAY TRACING LEVEL", config);
	this->exitButton = new Button("EXIT", config);
	
	// SelectView options
	rayTracingLevelView->addItem("OFF", 0);
	rayTracingLevelView->addItem("1", 1);
	rayTracingLevelView->addItem("2", 2);
	
	// Adding listeners
	pausedBox->addStateChangeAction(this);
	cullBackfaceBox->addStateChangeAction(this);
	wireframeBox->addStateChangeAction(this);
	fakeSizeBox->addStateChangeAction(this);
	starDimmerBox->addStateChangeAction(this);
	futureBodyPathBox->addStateChangeAction(this);
	futureBodyInputView->addIntegerInputAction(this);
	bodyLocatorBox->addStateChangeAction(this);
	bodyLocatorInputView->addIntegerInputAction(this);
	rayTracingLevelView->addSelectViewStateChangeAction(this);
	exitButton->addViewClickedAction(this);
	
	// Add view
	addChild(fpsView);
	addChild(timeView);
	addChild(pausedBox);
	addChild(cullBackfaceBox);
	addChild(wireframeBox);
	addChild(fakeSizeBox);
	addChild(starDimmerBox);
	addChild(futureBodyPathBox);
	addChild(futureBodyInputView);
	addChild(bodyLocatorBox);
	addChild(bodyLocatorInputView);
	addChild(rayTracingLevelView);
	addChild(exitButton);
}

MainHudPage::~MainHudPage(){
	// Debug
	if((debugLevel & 0x10) == 16){			
		std::cout << "MainHudPage.cpp\t\tFinalizing" << std::endl;
	}
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
	}else if(box == fakeSizeBox){ // Toggle fakeSize
		std::vector<Body*> *bodies = simulator->getBodies();
		int size = bodies->size();
	
		for(int i=0; i<size; i++){
			(*bodies)[i]->setFakeSize(newState);
		}
	}else if(box == starDimmerBox){ // Toggle star dimming
		simulator->getStarDimmer()->setActivated(newState);
	}else if(box == futureBodyPathBox){ // Toggle body path visualization
		simulator->getBodyTracer()->setActive(newState);
	}else if(box == bodyLocatorBox){ // Toggle body locator visualization
		simulator->getBodyLocator()->setActive(newState);
	}else if(box == pausedBox){ // Pause / Unpause the simulation
		simulator->setPaused(newState);
	}else if(box == cullBackfaceBox){
		if(newState){
			glEnable(GL_CULL_FACE);
		}else{
			glDisable(GL_CULL_FACE);
		}
	}
}

void MainHudPage::onStateChange(SelectView<int> *view, int i){
	simulator->getRayTracerSimulator()->setLevel(i);
}

void MainHudPage::onIntegerInput(IntegerInputView *view, int value){
	if(view == futureBodyInputView){	
		simulator->getBodyTracer()->calculateFuturePath(value);
	}else if(view == bodyLocatorInputView){
		simulator->getBodyLocator()->locateBody(value);
	}
}

void MainHudPage::draw(DrawService *drawService){
	// SUper
	HudPage::draw(drawService);
	
	// Updating time
	std::string date;
	std::string text = "DATE ";
	DateService::getDate(simulator->getTime(), &date);	
	text += date;
	this->timeView->setText(text);
	
	// Updating FPS
	text = "FPS ";
	text += std::to_string(simulator->getFrame()->getFPS());	
	this->fpsView->setText(text);
}
