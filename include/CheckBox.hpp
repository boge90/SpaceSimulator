#ifndef CHECK_BOX_H
#define CHECK_BOX_H

#include "../include/Button.hpp"
#include "../include/ViewClickedAction.hpp"

class CheckBox: public Button, public ViewClickedAction{
	private:
		bool state;
		unsigned char red;
		unsigned char green;
		unsigned char blue;
	
	public:
		/**
		* Creates an checkbox object
		**/
		CheckBox(std::string text);
		
		/**
		* Finalizes the checkbox object
		**/
		~CheckBox();
		
		/**
		* Returns the state of the checkbox object
		**/
		bool getState(void);
		
		/**
		* Called when the view has been clicked
		**/
		void viewClicked(View *view, int button, int action);
		
		/**
		* Called when the view is drawn
		**/
		void draw(DrawService *drawService);
};

#endif
