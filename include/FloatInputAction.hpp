#ifndef FLOAT_INPUT_VIEW_ACTION_H
#define FLOAT_INPUT_VIEW_ACTION_H

#include "../include/FloatInputView.hpp"

class FloatInputAction{
	public:
		/**
		* Called when the user inputs an integer in the associated IntegerInputView
		**/
		virtual void onFloatInput(FloatInputView *view, double value) = 0;
};

#endif
