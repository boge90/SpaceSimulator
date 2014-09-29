#ifndef CAMERA_HUD_PAGE_H
#define CAMERA_HUD_PAGE_H

#include "../include/HudPage.hpp"
#include "../include/Config.hpp"
#include "../include/Simulator.hpp"
#include "../include/TextView.hpp"
#include "../include/FloatInputView.hpp"
#include "../include/FloatInputAction.hpp"

class CameraHudPage: public HudPage, FloatInputAction{
	private:
		// Data
		Simulator *simulator;
		
		// Misc
		bool initialized;
			
		// GUI
		FloatInputView *fovView;
	public:
		/**
		*
		**/
		CameraHudPage(int x, int y, int width, int height, int number, Simulator *simulator, Config *config);
		
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
};

#endif
