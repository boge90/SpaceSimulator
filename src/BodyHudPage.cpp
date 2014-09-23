#include "../include/BodyHudPage.hpp"
#include <iostream>

BodyHudPage::BodyHudPage(int x, int y, int width, int height, int number, Body *body): HudPage(x, y, width, height, number){
	// Debug
	std::cout << "BodyHudPage.cpp\t\tInitializing for body " << number << std::endl;
	
	// Init
	this->body = body;
	this->positionViewX = new TextView("");
	this->positionViewY = new TextView("");
	this->positionViewZ = new TextView("");
	this->velocityViewX = new TextView("");
	this->velocityViewY = new TextView("");
	this->velocityViewZ = new TextView("");
	
	// Adding child
	addChild(positionViewX);
	addChild(positionViewY);
	addChild(positionViewZ);
	addChild(velocityViewX);
	addChild(velocityViewY);
	addChild(velocityViewZ);
}

BodyHudPage::~BodyHudPage(){
	std::cout << "BodyHudPage.cpp\t\tFinalizing" << std::endl;
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
	glm::dvec3 velocity = body->getVelocity();
	newText = "VELOCITY X ";
	newText.append(std::to_string(velocity.x));
	velocityViewX->setText(newText);
	
	newText = "VELOCITY Y ";
	newText.append(std::to_string(velocity.y));
	velocityViewY->setText(newText);
	
	newText = "VELOCITY Z ";
	newText.append(std::to_string(velocity.z));
	velocityViewZ->setText(newText);

	// Super
	HudPage::draw(drawService);
}
