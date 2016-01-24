#include "../include/CameraHudPage.hpp"

#include "../include/BodyCameraControl.hpp"
#include "../include/OrbitCameraControl.hpp"
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
	
	// GUI
	this->freeCameraButton = new Button("FREE CAMERA", config);
	this->bodyCameraSelectView = new SelectView<AbstractCamera*>("BODY CAMERA", config);
	this->orbitCameraSelectView = new SelectView<AbstractCamera*>("ORBIT CAMERA", config);
	
	// Cameras
	std::vector<Body*> *bodies = simulator->getBodies();
	this->currentActive = 0;
	this->cameraControllers = new std::vector<AbstractCamera*>();
	
	// Reserving size for camera pointers
	cameraControllers->reserve((bodies->size()*2)+1);
	
	// Free camera
	cameraControllers->push_back(new FreeCameraControl(frame->getWindow(), frame, config));
	
	// Body cameras
	for(size_t i = 0; i<bodies->size(); i++){
		// Creating body camera
		BodyCameraControl *bodyCamera = new BodyCameraControl(frame->getWindow(), frame, (*bodies)[i], config);
		cameraControllers->push_back(bodyCamera);
		bodyCameraSelectView->addItem(bodyCamera->getCameraName(), bodyCamera);
		
		// Creating orbit camera
		OrbitCameraControl *orbitCamera = new OrbitCameraControl(frame->getWindow(), frame, (*bodies)[i], config);
		cameraControllers->push_back(orbitCamera);
		orbitCameraSelectView->addItem(orbitCamera->getCameraName(), orbitCamera);
	}
	
	// Setting active camera
	activeCamera = (*cameraControllers)[this->currentActive];
	activeCamera->setActive(true);
	
	// GUI listeners
	freeCameraButton->addViewClickedAction(this);
	bodyCameraSelectView->addSelectViewStateChangeAction(this);
	orbitCameraSelectView->addSelectViewStateChangeAction(this);
	fovView->addFloatInputAction(this);
	
	// Setting value of FOV view
	float fov = activeCamera->getFieldOfView();
	std::string text = "";
	text.append(std::to_string(fov));
	fovView->setInput(text);
	
	// This
	addChild(freeCameraButton);
	addChild(bodyCameraSelectView);
	addChild(orbitCameraSelectView);
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
	activeCamera->setFieldOfView(float(value));
}

AbstractCamera* CameraHudPage::getActivatedCamera(void){
	return activeCamera;
}

std::vector<AbstractCamera*>* CameraHudPage::getCameras(void){
	return cameraControllers;
}

void CameraHudPage::onStateChange(SelectView<AbstractCamera*> *view, AbstractCamera *t){
	// Deactivating previous camera 
	activeCamera->setActive(false);
	
	// Setting new camera
	activeCamera = t;
	
	// Activating new camera
	activeCamera->setActive(true);
}

void CameraHudPage::onClick(View *view, int button, int action){
	// Deactivating previous camera 
	activeCamera->setActive(false);
	
	activeCamera = (*cameraControllers)[0]; // Free camera controller
	
	// Activating new camera
	activeCamera->setActive(true);
}
