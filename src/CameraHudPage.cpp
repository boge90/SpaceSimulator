#include "../include/CameraHudPage.hpp"
#include "../include/BodyCameraControl.hpp"
#include "../include/FreeCameraControl.hpp"

#include "../include/SelectView.hpp"

#include <iostream>

CameraHudPage::CameraHudPage(int x, int y, int width, int height, Simulator *simulator, Frame *frame, Config *config): HudPage(x, y, width, height, "CAMERA", config){
	// Debug
	if((debugLevel & 0x10) == 16){
		std::cout << "CameraHudPage.cpp\tInitializing" << std::endl;
	}
	
	// Init
	this->fovView = new FloatInputView("FIELD OF VIEW ", config);
	
	// Cameras
	std::vector<Body*> *bodies = simulator->getBodies();
	this->currentActive = 0;
	this->cameraControllers = new std::vector<AbstractCamera*>();
	
	// Reserving size for camera pointers
	cameraControllers->reserve(bodies->size()+1);
	
	// Free camera
	cameraControllers->push_back(new FreeCameraControl(frame->getWindow(), frame->getWidth(), frame->getHeight(), config));
	for(size_t i = 0; i<bodies->size(); i++){
		// Creating camera
		BodyCameraControl *controller = new BodyCameraControl(frame->getWindow(), frame, (*bodies)[i], config);
		cameraControllers->push_back(controller);
	}
	activeCamera = (*cameraControllers)[this->currentActive];
	activeCamera->setActive(true);
	
	// Select view
	SelectView<AbstractCamera*> *view = new SelectView<AbstractCamera*>("CAMERA", config);
	for(size_t i=0; i<cameraControllers->size(); i++){	
		AbstractCamera *camera = (*cameraControllers)[i];
		view->addItem(camera->getCameraName(), camera);
	}
	view->addSelectViewStateChangeAction(this);
	
	
	// Listeners
	fovView->addFloatInputAction(this);
	
	// Setting value of FOV view
	float fov = activeCamera->getFieldOfView();
	std::string text = "";
	text.append(std::to_string(fov));
	fovView->setInput(text);
	
	// This
	addChild(view);
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
}

void CameraHudPage::draw(DrawService *drawService){
	// Super
	HudPage::draw(drawService);
}

void CameraHudPage::onFloatInput(FloatInputView *view, double value){
	activeCamera->setFieldOfView(value);
}

AbstractCamera* CameraHudPage::getActivatedCamera(void){
	return activeCamera;
}

std::vector<AbstractCamera*>* CameraHudPage::getCameras(void){
	return cameraControllers;
}

void CameraHudPage::onStateChange(SelectView<AbstractCamera*> *view, AbstractCamera *t){
	activeCamera = t;
}
