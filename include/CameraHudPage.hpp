#ifndef CAMERA_HUD_PAGE_H
#define CAMERA_HUD_PAGE_H

class CameraHudPage;

#include "../include/HudPage.hpp"
#include "../include/Config.hpp"
#include "../include/Simulator.hpp"
#include "../include/TextView.hpp"
#include "../include/FloatInputView.hpp"
#include "../include/FloatInputAction.hpp"
#include "../include/SelectViewStateChangeAction.hpp"

class CameraHudPage: public HudPage, FloatInputAction, ViewClickedAction, SelectViewStateChangeAction<AbstractCamera*>{
	private:
		// Camera data
		std::vector<AbstractCamera*> *cameraControllers;
		AbstractCamera *activeCamera;
		unsigned int currentActive;
			
		// GUI
		Button *freeCameraButton;
		SelectView<AbstractCamera*> *bodyCameraSelectView;
		SelectView<AbstractCamera*> *orbitCameraSelectView;
		FloatInputView *fovView;
	public:
		/**
		*
		**/
		CameraHudPage(int x, int y, int width, int height, Simulator *simulator, Frame *frame, Config *config);
		
		/**
		*
		**/
		~CameraHudPage(void);
		
		/**
		* Called when the view is drawn
		**/
		void draw(DrawService *drawService);
		
		/**
		*
		**/
		void onFloatInput(FloatInputView *view, double value);
        
        /**
        * Returns the camera activated
        **/
        AbstractCamera* getActivatedCamera(void);
        
        /**
        * 
        **/
        std::vector<AbstractCamera*>* getCameras(void);
        
        /**
        *
        **/
        void onStateChange(SelectView<AbstractCamera*> *view, AbstractCamera *t);
        
        /**
		*
		**/
		void onClick(View *view, int button, int action);
};

#endif
