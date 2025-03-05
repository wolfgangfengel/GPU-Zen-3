#include "OnesweepSort.h"
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

OnesweepSort::OnesweepSort( oroDevice device, OrochiUtils& oroutils, oroStream stream ) : m_device{ device }, m_oroutils{ oroutils }
{
	oroGetDeviceProperties( &m_props, device );
	compileKernels();
	configure( stream );
}

void OnesweepSort::compileKernels() noexcept
{
	const std::string kernelPath = "../radixsort/RadixSortKernels.h";
	const std::string kernelIncludeDir = "../radixsort/";

	const auto includeArg{ "-I" + kernelIncludeDir };
	std::vector<const char*> opts;
	opts.push_back( includeArg.c_str() );

	// opts.push_back( "-g" );
	// opts.push_back( "--gpu-architecture=compute_70" );

#define LOAD_FUNC( var, kernel ) var = m_oroutils.getFunctionFromFile( m_device, kernelPath.c_str(), kernel, &opts );
	LOAD_FUNC( m_count, "onesweep_count" );
	LOAD_FUNC( m_reorderKey, "onesweep_reorderKey" );
	LOAD_FUNC( m_reorderKeyPair, "onesweep_reorderKeyPair" );
#undef LOAD_FUNC
}

void OnesweepSort::configure( oroStream stream ) noexcept
{
	u64 gpSumBuffer = sizeof( u32 ) * BIN_SIZE * sizeof( u32 /* key type */ );
	m_gpSumBuffer.resizeAsync( gpSumBuffer, false /*copy*/, stream );

	u64 lookBackBuffer = sizeof( u64 ) * ( BIN_SIZE * LOOKBACK_TABLE_SIZE );
	m_lookbackBuffer.resizeAsync( lookBackBuffer, false /*copy*/, stream );

	m_tailIterator.resizeAsync( 1, false /*copy*/, stream );
	m_tailIterator.resetAsync( stream );
	m_gpSumCounter.resizeAsync( 1, false /*copy*/, stream );
}

void OnesweepSort::sort( const KeyValueSoA& elementsToSort, const KeyValueSoA& tmp, uint32_t n, int startBit, int endBit, oroStream stream, Callback callback ) noexcept
{
	bool keyPair = elementsToSort.value != nullptr;

	int nIteration = div_round_up64( endBit - startBit, 8 );
	uint64_t numberOfBlocks = div_round_up64( n, RADIX_SORT_BLOCK_SIZE );

	// Buffers
	void* gpSumBuffer = m_gpSumBuffer.ptr();
	void* lookBackBuffer = m_lookbackBuffer.ptr();
	void* tailIteratorBuffer = m_tailIterator.ptr();

	callback.kernelBeg( "clear buffer", -1 );

	m_lookbackBuffer.resetAsync( stream );
	m_gpSumCounter.resetAsync( stream );
	m_gpSumBuffer.resetAsync( stream );

	callback.kernelEnd( "clear buffer", -1 );

	callback.kernelBeg( "count", -1 );

	{
		void* counter = m_gpSumCounter.ptr();
		int maxBlocksPerMP = 0;
		oroError e = oroModuleOccupancyMaxActiveBlocksPerMultiprocessor( &maxBlocksPerMP, m_count, ONESWEEP_COUNT_THREADS_PER_BLOCK, 0 );
		const int nBlocks = e == oroSuccess ? maxBlocksPerMP * m_props.multiProcessorCount : 2048;

		const void* args[] = { &elementsToSort.key, &n, &gpSumBuffer, &startBit, &counter };
		OrochiUtils::launch1D( m_count, nBlocks * ONESWEEP_COUNT_THREADS_PER_BLOCK, args, ONESWEEP_COUNT_THREADS_PER_BLOCK, 0, stream );
	}

	callback.kernelEnd( "count", -1 );

	auto s = elementsToSort;
	auto d = tmp;
	for( int i = 0; i < nIteration; i++ )
	{
		callback.kernelBeg( "reorder", i );

		if( keyPair )
		{
			const void* args[] = { &s.key, &d.key, &s.value, &d.value, &n, &gpSumBuffer, &lookBackBuffer, &tailIteratorBuffer, &startBit, &i };
			OrochiUtils::launch1D( m_reorderKeyPair, numberOfBlocks * REORDER_NUMBER_OF_THREADS_PER_BLOCK, args, REORDER_NUMBER_OF_THREADS_PER_BLOCK, 0, stream );
		}
		else
		{
			const void* args[] = { &s.key, &d.key, &n, &gpSumBuffer, &lookBackBuffer, &tailIteratorBuffer, &startBit, &i };
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
} // namespace Radixsort