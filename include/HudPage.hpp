#ifndef HUD_PAGE_H
#define HUD_PAGE_H

#include "../include/ListLayout.hpp"
#include "../include/TextView.hpp"
#include "../include/Config.hpp"

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
		HudPage(int x, int y, int width, int height, int number, Config *config);
		
		/**
		* Finalizes the HUD page object
		**/
		~HudPage(void);
};

#endif
