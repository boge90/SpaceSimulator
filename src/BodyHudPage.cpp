#include "../include/BodyHudPage.hpp"
#include <iostream>

BodyHudPage::BodyHudPage(int x, int y, int width, int height, std::string title, Body *body, Config *config): HudPage(x, y, width, height, title, config){
	// Debug
	if((debugLevel & 0x10) == 16){		
		std::cout << "BodyHudPage.cpp\t\tInitializing for " << title << std::endl;
	}
	
	// Init
	this->body = body;
	this->positionViewX = new TextView("", config);
	this->positionViewY = new TextView("", config);
	this->positionViewZ = new TextView("", config);
	this->velocityView = new TextView("", config);
	this->wireframeBox = new CheckBox("WIREFRAME", false, config);
	this->visualizationTypeView = new SelectView<Visualization>("VISUALIZATION TYPE", config);
	
	// Listeners
	wireframeBox->addStateChangeAction(this);
	visualizationTypeView->addSelectViewStateChangeAction(this);
	
	// Adding visualization types
	visualizationTypeView->addItem("NORMAL", NORMAL);
	visualizationTypeView->addItem("TEMPERATURE", TEMPERATURE);
	
	// Adding child
	addChild(positionViewX);
	addChild(positionViewY);
	addChild(positionViewZ);
	addChild(velocityView);
	addChild(wireframeBox);
	addChild(visualizationTypeView);
}

BodyHudPage::~BodyHudPage(){
	if((debugLevel & 0x10) == 16){
		std::cout << "BodyHudPage.cpp\t\tFinalizing" << std::endl;
	}
}

void BodyHudPage::onStateChange(CheckBox *box, bool newState){
	if( box == wireframeBox ){
		body->setWireframeMode(newState);
	}
}

void BodyHudPage::onStateChange(SelectView<Visualization> *view, Visualization t){
	body->setVisualizationType(t);
}

void BodyHudPage::draw(DrawService *drawService){
	// Updating center
	glm::dvec3 center = body->getCenter();
	std::string newText = "POSITION X ";
	newText.append(std::to_string(center.x));
	positionViewX->setText(newText);
	
	newText = "POSITION Y ";
	newText.append(std::to_string(center.y));
	positionViewY->setText(newText);
	
	newText = "POSITION Z ";
	newText.append(std::to_string(center.z));
	positionViewZ->setText(newText);
	
	// Updating velocity
	glm::dvec3 v = body->getVelocity();
	double vel = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
	newText = "VELOCITY ";
	newText.append(std::to_string(vel));
	velocityView->setText(newText);

	// Super
	HudPage::draw(drawService);
}
