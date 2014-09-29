#include "../include/Menu.hpp"
#include <iostream>
#include <vector>

Menu::Menu(GLFWwindow *window, Simulator *simulator, Config *config){
	// Debug
	this->debugLevel = config->getDebugLevel();
	if((debugLevel & 0x10) == 16){	
		std::cout << "Menu.cpp\t\tInitializing\n";
	}

	// Init
	std::vector<Body*> *bodies = simulator->getBodies();
	this->currentActive = 0;
	this->cameraControllers = new std::vector<AbstractCamera*>();
	
	// Setting the number of controllers
	cameraControllers->reserve(bodies->size()+1);
	
	// Initializing body controllers
	cameraControllers->push_back(new FreeCameraControl(window, simulator->getFrame()->getWidth(), simulator->getFrame()->getHeight(), config));
	for(size_t i = 0; i<bodies->size(); i++){
		BodyCameraControl *controller = new BodyCameraControl(window, simulator->getFrame(), (*bodies)[i], config);
		cameraControllers->push_back(controller);
	}
	
	// Setting initial active camera
	activeCamera = (*cameraControllers)[this->currentActive];
	
	// Setting active camera active
	activeCamera->setActive(true);
	
	// Need to intialize HUD after the camera has been activated
	this->hud = new HUD(window, simulator, config);
}

Menu::~Menu(void){
	if((debugLevel & 0x10) == 16){	
		std::cout << "Menu.cpp\t\tFinalizing\n";
	}
	
	for(size_t i = 0; i<cameraControllers->size(); i++){
		delete (*cameraControllers)[i];
	}
	
	delete hud;
	delete cameraControllers;
}

void Menu::render(void){
	hud->render();
}

AbstractCamera* Menu::getActivatedCamera(void){
	return activeCamera;
}

std::vector<AbstractCamera*>* Menu::getCameras(void){
	return cameraControllers;
}

void Menu::changeCamera(bool next){
	// Deactivating current camera
	activeCamera->setActive(false);

	if(next && currentActive < cameraControllers->size()-1){currentActive++;}
	else if(!next && currentActive > 0){currentActive--;}

	activeCamera = (*cameraControllers)[currentActive];	
	activeCamera->setActive(true);
}

void Menu::menuClicked(int button, int action, int x, int y){
	hud->hudClicked(button, action, x, y);
}

void Menu::toggleHUD(void){
	hud->toggleVisibility();
	
	// Reactivating camera after HUD is hidden, such that DELTA timing
	// used in cameras are not WAY too high
	if(!isHudVisible()){
		activeCamera->setActive(true);
	}
}

bool Menu::isHudVisible(void){
	return hud->isVisible();
}
