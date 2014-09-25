#include "../include/MpiHelper.hpp"

#include <iostream>
#include "/usr/include/mpi/mpi.h"

bool MpiHelper::init(int argc, char **args, Config *config){
	if((config->getDebugLevel() & 0x10) == 16){
		std::cout << "MpiHelper.cpp\t\tInitializing" << std::endl;
	}
	
	MPI_Init(&argc, &args);
	MPI_Comm_size(MPI_COMM_WORLD, config->getMpiSizePtr());
	MPI_Comm_rank(MPI_COMM_WORLD, config->getMpiRankPtr());
	
	return true;
}
