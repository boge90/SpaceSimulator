#ifndef CHECK_BOX_STATE_CHANGE_ACTION_H
#define CHECK_BOX_STATE_CHANGE_ACTION_H

#include "../include/CheckBox.hpp"

class CheckBoxStateChangeAction{
	public:
		virtual void onStateChange(CheckBox *box, bool newState) = 0;
};

#endif
