#ifndef VIEW_CLICKED_ACTION_H
#define VIEW_CLICKED_ACTION_H

class ViewClickedAction;

#include "../include/View.hpp"

class ViewClickedAction{
	public:
		/**
		*
		**/
		ViewClickedAction(){};
		
		/**
		*
		**/
		~ViewClickedAction(void){};
	
		/**
		*
		**/
		virtual void viewClicked(View *view, int button, int action) = 0;
};

#endif
