#ifndef BODY_HUD_PAGE_H
#define BODY_HUD_PAGE_H

#include "../include/HudPage.hpp"
#include "../include/Body.hpp"
#include "../include/Config.hpp"
#include "../include/CheckBox.hpp"
#include "../include/CheckBoxStateChangeAction.hpp"
#include "../include/SelectViewStateChangeAction.hpp"

class BodyHudPage: public HudPage, public CheckBoxStateChangeAction, SelectViewStateChangeAction<Visualization>{
	private:
		// Data
		Body *body;
		
		// GUI
		TextView *positionViewX;
		TextView *positionViewY;
		TextView *positionViewZ;
		TextView *velocityView;
		CheckBox *wireframeBox;
		
		SelectView<Visualization> *visualizationTypeView;
	public:
		/**
		*
		**/
		BodyHudPage(int x, int y, int width, int height, std::string title, Body *body, Config *config);
		
		/**
		*
		**/
		~BodyHudPage();
		
		/**
		* CheckBox state change listener
		**/
		void onStateChange(CheckBox *box, bool newState);
        
        /**
        *
        **/
        void onStateChange(SelectView<Visualization> *view, Visualization t);
		
		/**
		*
		**/
		void draw(DrawService *drawService);
};

#endif
