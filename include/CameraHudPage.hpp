#ifndef CAMERA_HUD_PAGE_H
#define CAMERA_HUD_PAGE_H

class CameraHudPage;

#include "../include/HudPage.hpp"
#include "../include/Config.hpp"
#include "../include/Simulator.hpp"
#include "../include/TextView.hpp"
#include "../include/FloatInputView.hpp"
#include "../include/FloatInputAction.hpp"
#include "../include/ViewClickedAction.hpp"

class CameraHudPage: public HudPage, FloatInputAction, ViewClickedAction{
	private:
		// Camera data
		std::vector<AbstractCamera*> *cameraControllers;
		AbstractCamera *activeCamera;
		unsigned int currentActive;
			
		// GUI
		FloatInputView *fovView;
		std::vector<Button*> *cameraButtons;
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
		*
		**/
		void onClick(View *view, int button, int action);
        
        /**
        * Returns the camera activated
        **/
        AbstractCamera* getActivatedCamera(void);
        
        /**
        * 
        **/
        std::vector<AbstractCamera*>* getCameras(void);
};

#endif
