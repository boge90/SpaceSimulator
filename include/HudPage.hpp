#ifndef HUD_PAGE_H
#define HUD_PAGE_H

#include "../include/ListLayout.hpp"
#include "../include/TextView.hpp"

class HudPage: public ListLayout{
	private:
		// Misc data
		int number;
		
		// GUI
		TextView *numberView;
		
	public:
		/**
		* Creates an empty HUD page object
		**/
		HudPage(int x, int y, int width, int height, int number);
		
		/**
		* Finalizes the HUD page object
		**/
		~HudPage(void);
		
		/**
		* Called when the HUD is updated
		**/
		void draw(DrawService *service);
};

#endif
