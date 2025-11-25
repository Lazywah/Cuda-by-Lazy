#include <iostream>
#include "GPU_bits.cuh"
#include "GPU_char_global.cuh"
#include "GPU_char_shared.cuh"

int main(int argc, char** argv)
{
	bits_hammingDistance_noPrint(argc, argv);

	bits_hammingDistance(argc, argv);
	std::cout << "\n\n----------------------------------------\n\n";
	char_hammingDistance_global(argc, argv);
	std::cout << "\n\n----------------------------------------\n\n";
	char_hammingDistance_shared(argc, argv);
}