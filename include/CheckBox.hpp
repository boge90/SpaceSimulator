#ifndef CHECK_BOX_H
#define CHECK_BOX_H

class CheckBox;

#include "../include/Button.hpp"
#include "../include/ViewClickedAction.hpp"
#include "../include/CheckBoxStateChangeAction.hpp"

#include <vector>

class CheckBox: public Button, public ViewClickedAction{
	private:
		bool state;
		unsigned char red;
		unsigned char green;
		unsigned char blue;
		
		// Listeners
		std::vector<CheckBoxStateChangeAction*> *listeners;
	public:
		/**
		* Creates an checkbox object
		**/
		CheckBox(std::string text, bool state);
		
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
		void onClick(View *view, int button, int action);
		
		/**
		* Called when the view is drawn
		**/
		void draw(DrawService *drawService);
		
		/**
		* Adds a state change listener
		**/
		void addStateChangeAction(CheckBoxStateChangeAction *action);
};

#endif
