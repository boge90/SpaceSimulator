#ifndef BODY_HUD_PAGE_H
#define BODY_HUD_PAGE_H

#include "../include/HudPage.hpp"
#include "../include/Body.hpp"
#include "../include/Config.hpp"

class BodyHudPage: public HudPage{
	private:
		// Data
		Body *body;
		
		// GUI
		TextView *positionViewX;
		TextView *positionViewY;
		TextView *positionViewZ;
		TextView *velocityViewX;
		TextView *velocityViewY;
		TextView *velocityViewZ;
	public:
		/**
		*
		**/
		BodyHudPage(int x, int y, int width, int height, int number, Body *body, Config *config);
		
		/**
		*
		**/
		~BodyHudPage();
		
		/**
		*
		**/
		void draw(DrawService *drawService);
};

#endif
