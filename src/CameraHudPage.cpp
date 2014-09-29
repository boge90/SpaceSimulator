#include "../include/CameraHudPage.hpp"
#include "../include/BodyCameraControl.hpp"
#include "../include/FreeCameraControl.hpp"

#include <iostream>

CameraHudPage::CameraHudPage(int x, int y, int width, int height, Simulator *simulator, Frame *frame, Config *config): HudPage(x, y, width, height, "CAMERA", config){
	// Debug
	if((debugLevel & 0x10) == 16){
		std::cout << "CameraHudPage.cpp\tInitializing" << std::endl;
	}
	
	// Init
	this->fovView = new FloatInputView("FIELD OF VIEW ", config);
	this->cameraButtons = new std::vector<Button*>();
	
	// Cameras
	std::vector<Body*> *bodies = simulator->getBodies();
	this->currentActive = 0;
	this->cameraControllers = new std::vector<AbstractCamera*>();
	
	// Reserving size for camera pointers
	cameraButtons->reserve(bodies->size()+1);
	cameraControllers->reserve(bodies->size()+1);
	
	// Free camera
	cameraControllers->push_back(new FreeCameraControl(frame->getWindow(), frame->getWidth(), frame->getHeight(), config));
	cameraButtons->push_back(new Button("FREE CAMERA", config));
	(*cameraButtons)[0]->addViewClickedAction(this);
	addChild((*cameraButtons)[0]);
	
	
	for(size_t i = 0; i<bodies->size(); i++){
		// Camera button text
		std::string text = *((*bodies)[i]->getName());
		text.append(" CAMERA");
	
		// Adding camera button to list and GUI, + add listener
		cameraButtons->push_back(new Button(text, config));
		(*cameraButtons)[i+1]->addViewClickedAction(this);
		addChild((*cameraButtons)[i+1]);
	
		// Creating camera
		BodyCameraControl *controller = new BodyCameraControl(frame->getWindow(), frame, (*bodies)[i], config);
		cameraControllers->push_back(controller);
	}
	activeCamera = (*cameraControllers)[this->currentActive];
	activeCamera->setActive(true);
	
	// Listeners
	fovView->addFloatInputAction(this);
	
	// Setting value of FOV view
	float fov = activeCamera->getFieldOfView();
	std::string text = "";
	text.append(std::to_string(fov));
	fovView->setInput(text);
	
	// This
	addChild(fovView);
}

CameraHudPage::~CameraHudPage(void){
	if((debugLevel & 0x10) == 16){
		std::cout << "CameraHudPage.cpp\tFinalizing" << std::endl;
	}
	
	// Freeing cameras
	for(size_t i = 0; i<cameraControllers->size(); i++){
		delete (*cameraControllers)[i];
	}
	delete cameraControllers;
	delete cameraButtons;
}

void CameraHudPage::draw(DrawService *drawService){
	// Super
	HudPage::draw(drawService);
}

void CameraHudPage::onFloatInput(FloatInputView *view, double value){
	activeCamera->setFieldOfView(value);
}

void CameraHudPage::onClick(View *view, int button, int action){
	// Deactivating current camera
	activeCamera->setActive(false);
	
	for(size_t i=0; i<cameraButtons->size(); i++){
		if((*cameraButtons)[i] == view){
			activeCamera = (*cameraControllers)[i];	
			activeCamera->setActive(true);
			
			float fov = activeCamera->getFieldOfView();
			std::string text = "";
			text.append(std::to_string(fov));
			fovView->setInput(text);
			break;
		}
	}
}

AbstractCamera* CameraHudPage::getActivatedCamera(void){
	return activeCamera;
}

std::vector<AbstractCamera*>* CameraHudPage::getCameras(void){
	return cameraControllers;
}
