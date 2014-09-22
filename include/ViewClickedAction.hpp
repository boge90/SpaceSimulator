#ifndef VIEW_CLICKED_ACTION_H
#define VIEW_CLICKED_ACTION_H

class ViewClickedAction;

#include "../include/View.hpp"

class ViewClickedAction{
	public:
		/**
		*
		**/
		virtual void onClick(View *view, int button, int action) = 0;
};

#endif
