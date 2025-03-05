#include "Orochi/Orochi.h"
#include <memory>
#include <stdint.h>
#include <vector>
#include <random>
#include <map>
#include <fstream>

#include "OnesweepSort.h"
#include "ClassicSort.h"

#include <Orochi/GpuMemory.h>

#include <ppl.h>

#include "json.hpp"

// Specify the GPU index
static int DEVICE_INDEX = 0;

struct splitmix64
{
	uint64_t x = 0; /* The state can be seeded with any value. */

	uint64_t next()
	{
		uint64_t z = ( x += 0x9e3779b97f4a7c15 );
		z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9;
		z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111eb;
		return z ^ ( z >> 31 );
	}
};


enum SORT_BITS
{
	SORT_BITS_8 = 8,
	SORT_BITS_16 = 16,
	SORT_BITS_24 = 24,
	SORT_BITS_32 = 32,
};

// return G elements per second.
template <class SortClass>
double measure( oroStream stream, SortClass& sortModule, int n, SORT_BITS bits, bool keyPair )
{
	std::vector<uint32_t> srcKey( n );
	splitmix64 rng;
	for( int i = 0; i < n; i++ )
	{
		uint32_t mask = (uint32_t)( ( 1ull << (uint64_t)bits ) - 1 );
		srcKey[i] = rng.next() & mask;
	}
	
	Oro::GpuMemory<uint32_t> keyGPU( n );
	Oro::GpuMemory<uint32_t> tmpKeyGPU( n );
	Oro::GpuMemory<uint32_t> valGPU( n );
	Oro::GpuMemory<uint32_t> tmpValGPU( n );

	Oro::GpuMemory<uint8_t> scanBuffer;
	if constexpr( std::is_same<SortClass, Radixsort::ClassicSort>() )
	{
		scanBuffer.resizeAsync( sortModule.scanBufferBytes( n ) );
	}

	std::vector<uint32_t> sortedKey = srcKey;
	// std::stable_sort( sortedKey.begin(), sortedKey.end() );
	concurrency::parallel_radixsort( sortedKey.begin(), sortedKey.end() );

	int nExec = 6;


	int nSample = 0;
	double totalSortTime = 0;

	for( int i = 0; i < nExec; i++ )
	{
		keyGPU.copyFromHost( srcKey.data(), n );
		valGPU.copyFromHost( srcKey.data(), n );

		OroStopwatch oroStream( stream );
		oroStream.start();

		SortClass::KeyValueSoA elementToSort = {};
		SortClass::KeyValueSoA tmp = {};
		elementToSort.key = keyGPU.ptr();
		tmp.key = tmpKeyGPU.ptr();
		if( keyPair )
		{
			elementToSort.value = valGPU.ptr();
			tmp.value = tmpValGPU.ptr();
		}

		if constexpr( std::is_same<SortClass, Radixsort::ClassicSort>() )
		{
			sortModule.sort( elementToSort, tmp, scanBuffer.ptr(), n, 0, bits, stream );
		}
		else
		{
			sortModule.sort( elementToSort, tmp, n, 0, bits, stream );
		}

		oroStream.stop();
		OrochiUtils::waitForCompletion();
		float ms = oroStream.getMs();
		float gKeys_s = static_cast<float>( n ) / 1000.f / 1000.f / ms;
		printf( "  %5.2fms (%3.2fGKeys/s) sorting %3.1fMkeys [%s]\n", ms, gKeys_s, n / 1000.f / 1000.f, keyPair ? "keyValue" : "key" );

		// skip head to wait until it gets stable
		if( 2 <= i )
		{
			totalSortTime += ms;
			nSample++;
		}

		// validate sort
		// The sort result is stored at the first arg of sort unlike orochi's module to avoid memory copy.
		std::vector<uint32_t> result = keyPair ? valGPU.getData() : keyGPU.getData(); 
		if( result != sortedKey )
		{
			abort();
		}
	}

	double ms = totalSortTime / nSample;
	double gKeys_s = static_cast<float>( n ) / 1000.f / 1000.f / ms;
	printf( "(%s) %5.2fms (%3.2fGKeys/s) sorting %3.1fMkeys [%s], %d bits, %d sample avg\n", typeid( SortClass ).name(), ms, gKeys_s, n / 1000.f / 1000.f, keyPair ? "keyValue" : "key", bits, nSample );

	return gKeys_s;
}

template <class SortClass>
void takeTrace( const char* tracefile, oroStream stream, SortClass& sortModule, int n, SORT_BITS bits )
{
	std::vector<uint32_t> srcKey( n );
	splitmix64 rng;
	for( int i = 0; i < n; i++ )
	{
		uint32_t mask = (uint32_t)( ( 1ull << (uint64_t)bits ) - 1 );
		srcKey[i] = rng.next() & mask;
	}
	Oro::GpuMemory<uint32_t> srcGPU( n );
	srcGPU.copyFromHost( srcKey.data(), n );

	Oro::GpuMemory<uint32_t> keyGPU( n );
	Oro::GpuMemory<uint32_t> tmpKeyGPU( n );

	Oro::GpuMemory<uint8_t> scanBuffer;
	if constexpr( std::is_same<SortClass, Radixsort::ClassicSort>() )
	{
		scanBuffer.resizeAsync( sortModule.scanBufferBytes( n ) );
	}

	std::vector<uint32_t> sortedKey = srcKey;
	// std::stable_sort( sortedKey.begin(), sortedKey.end() );
	concurrency::parallel_radixsort( sortedKey.begin(), sortedKey.end() );

	oroEvent eTraceBegin;
	oroEventCreateWithFlags( &eTraceBegin, oroEventDefault );
	oroEventRecord( eTraceBegin, stream );

	struct Interval
	{
		std::string name;
		oroEvent beg = 0;
		oroEvent end = 0;
		int pid = 0;
	};
	std::vector<Interval> intervals;

	SortClass::Callback callback;

	auto measureBeg = [&]( oroEvent* eBeg )
	{
		oroEventCreateWithFlags( eBeg, oroEventDefault );
		oroEventRecord( *eBeg, stream );
	};
	auto measureEnd = [&]( oroEvent eBeg, const char* name, int pid )
	{
		Interval interval;
		interval.name = name;
		interval.pid = pid;
		interval.beg = eBeg;
		oroEventCreateWithFlags( &interval.end, oroEventDefault );
		oroEventRecord( interval.end, stream );
		intervals.push_back( interval );
	};

	oroEvent kernelBeg = 0;
	int iterationCount = 0;

	callback.kernelBeg = [&]( const char* name, int i )
	{
		measureBeg( &kernelBeg );
	};
	callback.kernelEnd = [&]( const char* name, int i )
	{
		measureEnd( kernelBeg, name, i );
	};

	for( int i = 0; i < 4; i++ )
	{
		iterationCount = 0;

		oroMemcpyDtoDAsync( (oroDeviceptr)keyGPU.ptr(), (oroDeviceptr)srcGPU.ptr(), keyGPU.size() * sizeof( uint32_t ), stream );

		SortClass::KeyValueSoA elementToSort = {};
		SortClass::KeyValueSoA tmp = {};
		elementToSort.key = keyGPU.ptr();
		tmp.key = tmpKeyGPU.ptr();

		if constexpr( std::is_same<SortClass, Radixsort::ClassicSort>() )
		{
			sortModule.sort( elementToSort, tmp, scanBuffer.ptr(), n, 0, bits, stream, callback );
		}
		else
		{
			sortModule.sort( elementToSort, tmp, n, 0, bits, stream, callback );
		}

		// validate sort
		//std::vector<uint32_t> result = keyGPU.getData();
		//if( result != sortedKey )
		//{
		//	abort();
		//}
	}

	nlohmann::json traceEvents = nlohmann::json::array();
	for( auto interval : intervals )
	{
		oroEventSynchronize( interval.end );
		float begMS = 0;
		oroError ee = oroEventElapsedTime( &begMS, eTraceBegin, interval.beg );
		float durMS = 0;
		oroEventElapsedTime( &durMS, interval.beg, interval.end );

		nlohmann::json r;
		r["ph"] = "X";
		r["pid"] = interval.pid;
		r["ts"] = begMS * 1000.0;  // in microseconds
		r["dur"] = durMS * 1000.0; // in microseconds
		r["name"] = interval.name;

		traceEvents.push_back( r );
	}
	nlohmann::json chromeTracing = { 
		{ "traceEvents", traceEvents }
	};

	std::ofstream ofs( tracefile );
	ofs << chromeTracing;
}

int main()
{
	if( oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 ) )
	{
		printf( "failed to init..\n" );
		return 0;
	}

	oroError err;
	err = oroInit( 0 );
	oroDevice device;
	err = oroDeviceGet( &device, DEVICE_INDEX );
	oroCtx ctx;
	err = oroCtxCreate( &ctx, 0, device );
	oroCtxSetCurrent( ctx );

	oroStream stream = 0;
	oroStreamCreate( &stream );
	oroDeviceProp props;
	oroGetDeviceProperties( &props, device );

	bool isNvidia = oroGetCurAPI( 0 ) & ORO_API_CUDADRIVER;

	printf( "Device: %s\n", props.name );
	printf( "Cuda: %s\n", isNvidia ? "Yes" : "No" );

	OrochiUtils oroutils;

	Radixsort::OnesweepSort onesweep( device, oroutils );
	Radixsort::ClassicSort classicSort( device, oroutils );
	
	// Tracing
	uint32_t nElementsForOneshotMeasure = 1u << 27;
	printf( "=== tracing ===\n" );
	measure( stream, onesweep, nElementsForOneshotMeasure, SORT_BITS_32, false /*keyPair*/ );
	takeTrace( "trace_onesweep.json", stream, onesweep, nElementsForOneshotMeasure, SORT_BITS_32 );
	measure( stream, classicSort, nElementsForOneshotMeasure, SORT_BITS_32, false /*keyPair*/ );
	takeTrace( "trace_classic.json", stream, classicSort, nElementsForOneshotMeasure, SORT_BITS_32 );

	printf( "=== measurements as CSV ===\n" );
	{
		char name[256];
		sprintf( name, "perf_onesweep_%s.csv", props.name );
		FILE* fp = fopen( name, "w" );
		fprintf( fp, "n,G Elem/s\n" );

		for( int e = 16 ; e < 29 ; e++ )
		{
			int N_SPLIT = 16;
			for( int i = 0; i < N_SPLIT; i++ )
			{
				int n = pow( 2, e + ( 1.0 / N_SPLIT ) * i );

				double GKeys = measure( stream, onesweep, n, SORT_BITS_32, false /*keyPair*/ );
				fprintf( fp, "%d,%f\n", n, GKeys );
			}
		}
		fclose( fp );
	}

	printf( "=== batch measurements & validations ===\n" );
	for( auto sortBits : { SORT_BITS_32, SORT_BITS_24, SORT_BITS_16, SORT_BITS_8 } )
	{
		for( auto keyPair : { false, true } )
		{
			for( auto n : { 1u << 16, 1u << 27, 1u << 28, 1u << 29 } )
			{
				measure( stream, onesweep, n, sortBits, keyPair );
			}
		}
	}

	for( auto sortBits : { SORT_BITS_32, SORT_BITS_24, SORT_BITS_16, SORT_BITS_8 } )
	{
		for( auto keyPair : { false, true } )
		{
			for( auto n : { 1u << 16, 1u << 27, 1u << 28, 1u << 29 } )
			{
				measure( stream, classicSort, n, sortBits, keyPair );
			}
		}
	}

	printf( "=== corner cases ===\n" );
	for( auto sortBits : { SORT_BITS_32, SORT_BITS_24, SORT_BITS_16, SORT_BITS_8 } )
	{
		for( auto keyPair : { false, true } )
		{
			for( auto n : { 1u << 8, ( 1u << 10 ) + 5, ( 1u << 22 ) + 111 } )
			{
				measure( stream, classicSort, n, sortBits, keyPair );
			}
		}
	}

    oroCtxDestroy( ctx );

	return 0;
}