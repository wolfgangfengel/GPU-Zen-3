#include "ClassicSort.h"
#include "RadixSortConfigs.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>

namespace Radixsort
{

constexpr uint64_t div_round_up64( uint64_t val, uint64_t divisor ) noexcept { return ( val + divisor - 1 ) / divisor; }
constexpr uint64_t next_multiple64( uint64_t val, uint64_t divisor ) noexcept { return div_round_up64( val, divisor ) * divisor; }

ClassicSort::ClassicSort( oroDevice device, OrochiUtils& oroutils, oroStream stream ) : m_device{ device }, m_oroutils{ oroutils }
{
	compileKernels();
	configure( stream );
}

void ClassicSort::compileKernels() noexcept
{
	const std::string kernelPath = "../radixsort/RadixSortKernels.h";
	const std::string kernelIncludeDir = "../radixsort/";

	const auto includeArg{ "-I" + kernelIncludeDir };
	std::vector<const char*> opts;
	opts.push_back( includeArg.c_str() );
	// opts.push_back( "-G" );
	// opts.push_back( "-g" );
	// opts.push_back( "--gpu-architecture=compute_70" );

#define LOAD_FUNC( var, kernel ) var = m_oroutils.getFunctionFromFile( m_device, kernelPath.c_str(), kernel, &opts );
	LOAD_FUNC( m_count, "classic_counting" );
	LOAD_FUNC( m_scan, "classic_scan" );
	LOAD_FUNC( m_reorderKey, "classic_reorderKey" );
	LOAD_FUNC( m_reorderKeyPair, "classic_reorderKeyPair" );
#undef LOAD_FUNC
}

void ClassicSort::configure( oroStream stream ) noexcept
{
	m_scanIterator.resizeAsync( 1, false /*copy*/, stream );
}

inline uint32_t extractDigit( uint32_t x, uint32_t bitLocation ) { return ( x >> bitLocation ) & RADIX_MASK; }

void ClassicSort::sort( const KeyValueSoA& elementsToSort, const KeyValueSoA& tmp, void* scanBuffer, uint32_t n, int startBit, int endBit, oroStream stream, Callback callback ) noexcept
{
	bool keyPair = elementsToSort.value != nullptr;

	int nIteration = div_round_up64( endBit - startBit, 8 );
	uint64_t numberOfBlocks = div_round_up64( n, RADIX_SORT_BLOCK_SIZE );

	// Buffers

	auto s = elementsToSort;
	auto d = tmp;
	for( int i = 0; i < nIteration; i++ )
	{
		m_scanIterator.resetAsync( stream );

		// counting

		callback.kernelBeg( "count", i );

		u32 bitLocation = startBit + N_RADIX * i;
		{
			const void* args[] = { &s.key, &n, &scanBuffer, &bitLocation };
			OrochiUtils::launch1D( m_count, numberOfBlocks * CLASSIC_COUNT_THREADS_PER_BLOCK, args, CLASSIC_COUNT_THREADS_PER_BLOCK, 0, stream );
		}

		callback.kernelEnd( "count", i );

		// scan
		callback.kernelBeg( "scan", i );
		{
			u32 nScanInput = scanBufferBytes( n ) / 4;
			u32 numberOfScanBlock = div_round_up64( nScanInput, CLASSIC_SCAN_BLOCK_SIZE );

			const void* args[] = { &scanBuffer, &nScanInput, m_scanIterator.address() };
			OrochiUtils::launch1D( m_scan, numberOfScanBlock * CLASSIC_SCAN_THREADS_PER_BLOCK, args, CLASSIC_SCAN_THREADS_PER_BLOCK, 0, stream );
		}
		callback.kernelEnd( "scan", i );

		// reorder
		callback.kernelBeg( "reorder", i );
		if( keyPair )
		{
			const void* args[] = { &s.key, &d.key, &s.value, &d.value, &n, &scanBuffer, &startBit, &i };
			OrochiUtils::launch1D( m_reorderKeyPair, numberOfBlocks * REORDER_NUMBER_OF_THREADS_PER_BLOCK, args, REORDER_NUMBER_OF_THREADS_PER_BLOCK, 0, stream );
		}
		else
		{
			const void* args[] = { &s.key, &d.key, &n, &scanBuffer, &startBit, &i };
			OrochiUtils::launch1D( m_reorderKey, numberOfBlocks * REORDER_NUMBER_OF_THREADS_PER_BLOCK, args, REORDER_NUMBER_OF_THREADS_PER_BLOCK, 0, stream );
		}
		callback.kernelEnd( "reorder", i );

		std::swap( s, d );
	}

	if( s.key /* current output */ != elementsToSort.key )
	{
		oroMemcpyDtoDAsync( (oroDeviceptr)elementsToSort.key, (oroDeviceptr)tmp.key, sizeof( uint32_t ) * n, stream );

		if( keyPair )
		{
			oroMemcpyDtoDAsync( (oroDeviceptr)elementsToSort.value, (oroDeviceptr)tmp.value, sizeof( uint32_t ) * n, stream );
		}
	}
}

ClassicSort::u32 ClassicSort::scanBufferBytes( uint32_t n ) const
{
	uint64_t numberOfBlocks = div_round_up64( n, RADIX_SORT_BLOCK_SIZE );
	return sizeof( uint32_t ) * BIN_SIZE * numberOfBlocks;
}
} // namespace Radixsort