#ifndef MPI_HELPER_H
#define MPI_HELPER_H

#include "../include/Config.hpp"

class MpiHelper{
	public:
		static bool init(int argc, char **args, Config *config);
};

#endif
