#ifndef INTEGER_INPUT_VIEW_ACTION_H
#define INTEGER_INPUT_VIEW_ACTION_H

#include "../include/IntegerInputView.hpp"

class IntegerInputAction{
	public:
		/**
		* Called when the user inputs an integer in the associated IntegerInputView
		**/
		virtual void onIntegerInput(IntegerInputView *view, int value) = 0;
};

#endif
