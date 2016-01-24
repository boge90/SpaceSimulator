#ifndef COMBINED_BODY_HUD_PAGE_H
#define COMBINED_BODY_HUD_PAGE_H

#include "../include/HudPage.hpp"
#include "../include/Body.hpp"
#include "../include/Config.hpp"
#include "../include/CheckBox.hpp"
#include "../include/CheckBoxStateChangeAction.hpp"
#include "../include/SelectViewStateChangeAction.hpp"
#include <vector>

class CombinedBodyHudPage: public HudPage, public CheckBoxStateChangeAction, SelectViewStateChangeAction<Visualization>{
	private:
		// Data
		Body *body;
		
		// GUI
		std::vector<TextView*> *positionViews;
		std::vector<TextView*> *velocityViews;
		std::vector<CheckBox*> *wireframeBoxes;
		std::vector<SelectView<Visualization>*> *visualizationTypeViews;

		/* Data */
		std::vector<Body*> *bodies;
		
	public:
		/**
		*
		**/
		CombinedBodyHudPage(int x, int y, int width, int height, std::string title, std::vector<Body*> *bodies, Config *config);
		
		/**
		*
		**/
		~CombinedBodyHudPage();
		
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
