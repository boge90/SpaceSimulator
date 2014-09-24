#ifndef BUTTON_H
#define BUTTON_H

#include "../include/TextView.hpp"
#include "../include/Config.hpp"
#include <string>

class Button: public TextView{
	public:
		/**
		* Creates a button object
		**/
		Button(std::string text, Config *config);
		
		/**
		*
		**/
		~Button();
		
		/**
		* Called when the view has been clicked on
		**/
		virtual void clicked(int button, int action);
};

#endif
