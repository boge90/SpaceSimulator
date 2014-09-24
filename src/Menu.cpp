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
	this->hud = new HUD(window, simulator, config);
	this->currentActive = 0;
	freeCameraControl = new FreeCameraControl(window, simulator->getFrame()->getWidth(), simulator->getFrame()->getHeight(), config);
	std::vector<Body*> *bodies = simulator->getBodies();
	bodyCameraControllers = new std::vector<BodyCameraControl*>();
	
	// Setting the number of controllers
	bodyCameraControllers->reserve(bodies->size());
	
	// Initializing body controllers
	int size = bodies->size();
	for(int i = 0; i<size; i++){
		BodyCameraControl *controller = new BodyCameraControl(window, simulator->getFrame(), (*bodies)[i], config);
		bodyCameraControllers->push_back(controller);
	}
	
	// Setting initial active camera
	activeCamera = freeCameraControl;
}

Menu::~Menu(void){
	if((debugLevel & 0x10) == 16){	
		std::cout << "Menu.cpp\t\tFinalizing\n";
	}
	
	
	int size = bodyCameraControllers->size();
	for(int i = 0; i<size; i++){
		delete (*bodyCameraControllers)[i];
	}
	
	delete hud;
	delete freeCameraControl;
	delete bodyCameraControllers;
}

void Menu::render(void){
	hud->render();
}

AbstractCamera* Menu::getActivatedCamera(void){
	return activeCamera;
}

void Menu::changeCamera(bool next){
	if(next && currentActive < bodyCameraControllers->size()){currentActive++;}
	else if(!next && currentActive > 0){currentActive--;}

	if(currentActive == 0 && activeCamera != freeCameraControl){
		std::cout << "Menu.cpp\t\tChanging to Free camera\n";
		activeCamera = freeCameraControl;
	}else if(currentActive <= bodyCameraControllers->size() && currentActive > 0 && activeCamera != (*bodyCameraControllers)[currentActive-1]){
		std::cout << "Menu.cpp\t\tChanging to Body camera " << (currentActive-1) << "\n";
		activeCamera = (*bodyCameraControllers)[currentActive-1];
	}
	
	activeCamera->activated();
}

void Menu::menuClicked(int button, int action, int x, int y){
	hud->hudClicked(button, action, x, y);
}

void Menu::toggleHUD(void){
	hud->toggleVisibility();
	
	// Reactivating camera after HUD is hidden, such that DELTA timing
	// used in cameras are not WAY too high
	if(!isHudVisible()){
		activeCamera->activated();
	}
}

bool Menu::isHudVisible(void){
	return hud->isVisible();
}
