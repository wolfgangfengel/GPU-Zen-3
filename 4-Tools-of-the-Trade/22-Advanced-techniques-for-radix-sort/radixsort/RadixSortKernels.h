#include "RadixSortConfigs.h"

#if defined( CUDART_VERSION ) && CUDART_VERSION >= 9000
#define ITS 1
#endif

using namespace Radixsort;

namespace
{
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
} // namespace

using RADIX_SORT_KEY_TYPE = u32;
using RADIX_SORT_VALUE_TYPE = u32;

#if defined( DESCENDING_ORDER )
constexpr u32 ORDER_MASK_32 = 0xFFFFFFFF;
constexpr u64 ORDER_MASK_64 = 0xFFFFFFFFFFFFFFFFllu;
#else
constexpr u32 ORDER_MASK_32 = 0;
constexpr u64 ORDER_MASK_64 = 0llu;
#endif

__device__ constexpr u32 div_round_up( u32 val, u32 divisor ) noexcept { return ( val + divisor - 1 ) / divisor; }

template<int NElement, int NThread, class T>
__device__ void clearShared( T* sMem, T value )
{
	for( int i = 0; i < NElement; i += NThread )
	{
		if( i < NElement )
		{
			sMem[i + threadIdx.x] = value;
		}
	}
}

__device__ inline u32 getKeyBits( u32 x ) { return x ^ ORDER_MASK_32; }
__device__ inline u64 getKeyBits( u64 x ) { return x ^ ORDER_MASK_64; }
__device__ inline u32 extractDigit( u32 x, u32 bitLocation ) { return ( x >> bitLocation ) & RADIX_MASK; }
__device__ inline u32 extractDigit( u64 x, u32 bitLocation ) { return (u32)( ( x >> bitLocation ) & RADIX_MASK ); }

template <class T>
__device__ inline T scanExclusive( T prefix, T* sMemIO, int nElement )
{
	// assert(nElement <= nThreads)
	bool active = threadIdx.x < nElement;
	T value = active ? sMemIO[threadIdx.x] : 0;
	T x = value;

	for( u32 offset = 1; offset < nElement; offset <<= 1 )
	{
		if( active && offset <= threadIdx.x )
		{
			x += sMemIO[threadIdx.x - offset];
		}

		__syncthreads();

		if( active )
		{
			sMemIO[threadIdx.x] = x;
		}

		__syncthreads();
	}

	T sum = sMemIO[nElement - 1];

	__syncthreads();

	if( active )
	{
		sMemIO[threadIdx.x] = x + prefix - value;
	}

	__syncthreads();

	return sum;
}

extern "C" __global__ void onesweep_count( RADIX_SORT_KEY_TYPE* inputs, u32 numberOfInputs, u32* gpSumBuffer, u32 startBits, u32* counter )
{
	__shared__ u32 localCounters[sizeof( RADIX_SORT_KEY_TYPE )][BIN_SIZE];

	for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
	{
		for( int j = threadIdx.x; j < BIN_SIZE; j += ONESWEEP_COUNT_THREADS_PER_BLOCK )
		{
			localCounters[i][j] = 0;
		}
	}

	u32 numberOfBlocks = div_round_up( numberOfInputs, ONESWEEP_COUNT_ITEM_PER_BLOCK );
	__shared__ u32 iBlock;
	for(;;)
	{
		if( threadIdx.x == 0 )
		{
			iBlock = atomicInc( counter, 0xFFFFFFFF );
		}

		__syncthreads();

		if( numberOfBlocks <= iBlock )
			break;
    
		for( int j = 0; j < ONESWEEP_COUNT_ITEMS_PER_THREAD; j++ )
		{
			u32 itemIndex = iBlock * ONESWEEP_COUNT_ITEM_PER_BLOCK + threadIdx.x * ONESWEEP_COUNT_ITEMS_PER_THREAD + j;
			if( itemIndex < numberOfInputs )
			{
				auto item = inputs[itemIndex];
				for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
				{
					u32 bitLocation = startBits + i * N_RADIX;
					u32 bits = extractDigit( getKeyBits( item ), bitLocation );
					atomicInc( &localCounters[i][bits], 0xFFFFFFFF );
				}
			}
		}

		__syncthreads();
	}

	for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
	{
		scanExclusive<u32>( 0, &localCounters[i][0], BIN_SIZE );
	}

	for( int i = 0; i < sizeof( RADIX_SORT_KEY_TYPE ); i++ )
	{
		for( int j = threadIdx.x; j < BIN_SIZE; j += ONESWEEP_COUNT_THREADS_PER_BLOCK )
		{
			atomicAdd( &gpSumBuffer[BIN_SIZE * i + j], localCounters[i][j] );
		}
	}
}

template <bool keyPair>
__device__ __forceinline__ void onesweep_reorder( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, u32 numberOfInputs, u32* gpSumBuffer,
												  volatile u64* lookBackBuffer, u32* tailIterator, u32 startBits, u32 iteration )
{
	__shared__ u32 pSum[BIN_SIZE];

	struct SMem
	{
		struct Phase1
		{
			u16 blockHistogram[BIN_SIZE];
			u16 lpSum[BIN_SIZE * REORDER_NUMBER_OF_WARPS];
		};
		struct Phase2
		{
			RADIX_SORT_KEY_TYPE elements[RADIX_SORT_BLOCK_SIZE];
		};
		struct Phase3
		{
			RADIX_SORT_VALUE_TYPE elements[RADIX_SORT_BLOCK_SIZE];
			u8 buckets[RADIX_SORT_BLOCK_SIZE];
		};

		union
		{
			Phase1 phase1;
			Phase2 phase2;
			Phase3 phase3;
		} u;
	};
	__shared__ SMem smem;

	u32 bitLocation = startBits + N_RADIX * iteration;
	u32 blockIndex = blockIdx.x;
	u32 numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );

	clearShared<BIN_SIZE * REORDER_NUMBER_OF_WARPS, REORDER_NUMBER_OF_THREADS_PER_BLOCK, u16>( smem.u.phase1.lpSum, 0 );

	__syncthreads();

	RADIX_SORT_KEY_TYPE keys[REORDER_NUMBER_OF_ITEM_PER_THREAD];
	u32 localOffsets[REORDER_NUMBER_OF_ITEM_PER_THREAD];

	int warp = threadIdx.x / WARP_SIZE;
	int lane = threadIdx.x % WARP_SIZE;

	for( int i = 0, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i + lane;
		if( itemIndex < numberOfInputs )
		{
			keys[k] = inputKeys[itemIndex];
		}
	}
	for( int i = 0, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i + lane;
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );

		// check the attendees
		u32 broThreads =
#if defined( ITS )
			__ballot_sync( 0xFFFFFFFF,
#else
			__ballot(
#endif
						   itemIndex < numberOfInputs );

		for( int j = 0; j < N_RADIX; ++j )
		{
			u32 bit = ( bucketIndex >> j ) & 0x1;
			u32 difference = ( 0xFFFFFFFF * bit ) ^
#if defined( ITS )
								__ballot_sync( 0xFFFFFFFF, bit != 0 );
#else
								__ballot( bit != 0 );
#endif
			broThreads &= ~difference;
		}

		u32 lowerMask = ( 1u << lane ) - 1;
		auto digitCount = smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
		localOffsets[k] = digitCount + __popc( broThreads & lowerMask );
		
#if defined( ITS )
		__syncwarp( 0xFFFFFFFF );
#else
		__syncthreads();
#endif
		u32 leaderIdx = __ffs( broThreads ) - 1;
		if( lane == leaderIdx )
		{
			smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp] = digitCount + __popc( broThreads );
		}
#if defined( ITS )
		__syncwarp( 0xFFFFFFFF );
#else
		__syncthreads();
#endif
	}

	__syncthreads();

	for( int bucketIndex = threadIdx.x; bucketIndex < BIN_SIZE; bucketIndex += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = 0;
		for( int warp = 0; warp < REORDER_NUMBER_OF_WARPS; warp++ )
		{
			s += smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
		}
		smem.u.phase1.blockHistogram[bucketIndex] = s;
	}

	enum
	{
		STATUS_X = 0,
		STATUS_A,
		STATUS_P,
	};
	struct ParitionID
	{
		u64 value : 32;
		u64 block : 30;
		u64 flag : 2;
	};
	auto asPartition = []( u64 x )
	{
		ParitionID pa;
		memcpy( &pa, &x, sizeof( ParitionID ) );
		return pa;
	};
	auto asU64 = []( ParitionID pa )
	{
		u64 x;
		memcpy( &x, &pa, sizeof( u64 ) );
		return x;
	};

	if( threadIdx.x == 0 && LOOKBACK_TABLE_SIZE <= blockIndex )
	{
		// Wait until blockIndex < tail - MAX_LOOK_BACK + LOOKBACK_TABLE_SIZE
		while( ( atomicAdd( tailIterator, 0 ) & TAIL_MASK ) - MAX_LOOK_BACK + LOOKBACK_TABLE_SIZE <= blockIndex )
			;
	}
	__syncthreads();

	// A workaround for buffer clear on each iterations.
	u32 iterationBits = 0x20000000 * ( iteration & 0x1 );

	for( int i = threadIdx.x; i < BIN_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = smem.u.phase1.blockHistogram[i];
		int pIndex = BIN_SIZE * ( blockIndex % LOOKBACK_TABLE_SIZE ) + i;

		{
			ParitionID pa;
			pa.value = s;
			pa.block = blockIndex | iterationBits;
			pa.flag = STATUS_A;
			lookBackBuffer[pIndex] = asU64( pa );
		}

		u32 gp = gpSumBuffer[iteration * BIN_SIZE + i];

		u32 p = 0;

		for( int iBlock = (int)blockIndex - 1; 0 <= iBlock; iBlock-- )
		{
			int lookbackIndex = BIN_SIZE * ( iBlock % LOOKBACK_TABLE_SIZE ) + i;
			ParitionID pa;

			// when you reach to the maximum, flag must be STATUS_P(=0b10). flagRequire = 0b10
			// Otherwise, flag can be STATUS_A(=0b01) or STATUS_P(=0b10) flagRequire = 0b11
			int flagRequire = MAX_LOOK_BACK == blockIndex - iBlock ? STATUS_P : STATUS_A | STATUS_P;

			do
			{
				pa = asPartition( lookBackBuffer[lookbackIndex] );
			} while( ( pa.flag & flagRequire ) == 0 || pa.block != ( iBlock | iterationBits ) );

			u32 value = pa.value;
			p += value;
			if( pa.flag == STATUS_P )
			{
				break;
			}
		}

		ParitionID pa;
		pa.value = p + s;
		pa.block = blockIndex | iterationBits;
		pa.flag = STATUS_P;
		lookBackBuffer[pIndex] = asU64( pa );

		// complete global output location
		u32 globalOutput = gp + p;
		pSum[i] = globalOutput;
	}

	__syncthreads();

	if( threadIdx.x == 0 )
	{
		while( ( atomicAdd( tailIterator, 0 ) & TAIL_MASK ) != ( blockIndex & TAIL_MASK ) )
			;

		atomicInc( tailIterator, numberOfBlocks - 1 /* after the vary last item, it will be zero */ );
	}

	__syncthreads();

	u32 prefix = 0;
	for( int i = 0; i < BIN_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		prefix += scanExclusive<u16>( prefix, smem.u.phase1.blockHistogram + i, min( REORDER_NUMBER_OF_THREADS_PER_BLOCK, BIN_SIZE ) );
	}

	for( int bucketIndex = threadIdx.x; bucketIndex < BIN_SIZE; bucketIndex += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = smem.u.phase1.blockHistogram[bucketIndex];

		pSum[bucketIndex] -= s; // pre-substruct to avoid pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex] to calculate destinations

		for( int w = 0; w < REORDER_NUMBER_OF_WARPS; w++ )
		{
			int index = bucketIndex * REORDER_NUMBER_OF_WARPS + w;
			u32 n = smem.u.phase1.lpSum[index];
			smem.u.phase1.lpSum[index] = s;
			s += n;
		}
	}

	__syncthreads();

	for( int k = 0; k < REORDER_NUMBER_OF_ITEM_PER_THREAD; k++ )
	{
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
		localOffsets[k] += smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
	}

	__syncthreads();

	for( int i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
		if( itemIndex < numberOfInputs )
		{
			smem.u.phase2.elements[localOffsets[k]] = keys[k];
		}
	}

	__syncthreads();

	for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
		if( itemIndex < numberOfInputs )
		{
			auto item = smem.u.phase2.elements[i];
			u32 bucketIndex = extractDigit( getKeyBits( item ), bitLocation );

			// u32 dstIndex = pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex];
			u32 dstIndex = pSum[bucketIndex] + i;
			outputKeys[dstIndex] = item;
		}
	}

	if constexpr( keyPair )
	{
		__syncthreads();

		for( int i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
		{
			u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;
			u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
			if( itemIndex < numberOfInputs )
			{
				smem.u.phase3.elements[localOffsets[k]] = inputValues[itemIndex];
				smem.u.phase3.buckets[localOffsets[k]] = bucketIndex;
			}
		}

		__syncthreads();

		for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
		{
			u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
			if( itemIndex < numberOfInputs )
			{
				auto item       = smem.u.phase3.elements[i];
				u32 bucketIndex = smem.u.phase3.buckets[i];

				// u32 dstIndex = pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex];
				u32 dstIndex = pSum[bucketIndex] + i;
				outputValues[dstIndex] = item;
			}
		}
	}
}
extern "C" __global__ void __launch_bounds__( REORDER_NUMBER_OF_THREADS_PER_BLOCK ) onesweep_reorderKey( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, u32 numberOfInputs, u32* gpSumBuffer, volatile u64* lookBackBuffer, u32* tailIterator, u32 startBits,
												  u32 iteration )
{
	onesweep_reorder<false /*keyPair*/>( inputKeys, outputKeys, nullptr, nullptr, numberOfInputs, gpSumBuffer, lookBackBuffer, tailIterator, startBits, iteration );
}
extern "C" __global__ void __launch_bounds__( REORDER_NUMBER_OF_THREADS_PER_BLOCK ) onesweep_reorderKeyPair( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues,
																											   u32 numberOfInputs,
																								   u32* gpSumBuffer,
													  volatile u64* lookBackBuffer, u32* tailIterator, u32 startBits, u32 iteration )
{
	onesweep_reorder<true /*keyPair*/>( inputKeys, outputKeys, inputValues, outputValues, numberOfInputs, gpSumBuffer, lookBackBuffer, tailIterator, startBits, iteration );
}


extern "C" __global__ void classic_counting( RADIX_SORT_KEY_TYPE* inputs, u32 numberOfInputs, u32* scanBuffer, u32 bitLocation )
{
	__shared__ u32 localCounters[256];
	clearShared<BIN_SIZE, CLASSIC_COUNT_THREADS_PER_BLOCK, u32>( localCounters, 0 );
	__syncthreads();

	for( int i = 0; i < RADIX_SORT_BLOCK_SIZE; i += CLASSIC_COUNT_THREADS_PER_BLOCK )
	{
		u32 itemIndex = blockIdx.x * RADIX_SORT_BLOCK_SIZE + threadIdx.x + i;
		if( itemIndex < numberOfInputs )
		{
			auto item = inputs[itemIndex];
			u32 bucketIndex = extractDigit( getKeyBits( item ), bitLocation );
			atomicInc( &localCounters[bucketIndex], 0xFFFFFFFF );
		}
	}

	__syncthreads();

	u32 numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );
	for( int bucketIndex = threadIdx.x; bucketIndex < BIN_SIZE; bucketIndex += CLASSIC_COUNT_THREADS_PER_BLOCK )
	{
		u32 blockIndex = blockIdx.x;
		u32 counterIndex = bucketIndex * numberOfBlocks + blockIndex;
		scanBuffer[counterIndex] = localCounters[bucketIndex];
	}
}


extern "C" __global__ void classic_scan( u32* inout, u32 numberOfInputs, u64* g_iterator )
{
	__shared__ u32 gp;
	__shared__ u32 smem[CLASSIC_SCAN_THREADS_PER_BLOCK];

	u32 blockIndex = blockIdx.x;

	__syncthreads();

	u32 s = 0;
	for( int i = 0; i < CLASSIC_SCAN_BLOCK_SIZE; i += CLASSIC_SCAN_THREADS_PER_BLOCK )
	{
		u32 itemIndex = blockIndex * CLASSIC_SCAN_BLOCK_SIZE + i + threadIdx.x;
		s += itemIndex < numberOfInputs ? inout[itemIndex] : 0;
	}

	smem[threadIdx.x] = s;
	__syncthreads();

	// Parallel Reduction
	for( int i = 1; i < CLASSIC_SCAN_THREADS_PER_BLOCK; i *= 2 )
	{
		if( threadIdx.x < ( threadIdx.x ^ i ) )
			smem[threadIdx.x] += smem[threadIdx.x ^ i];
		__syncthreads();
	}

	if( threadIdx.x == 0 )
	{
		u32 prefix = smem[0];

		u64 expected;
		u64 cur = *g_iterator;
		u32 globalPrefix = cur & 0xFFFFFFFF;
		do
		{
			expected     = (u64)globalPrefix              | ( (u64)( blockIndex )     << 32 );
			u64 newValue = (u64)( globalPrefix + prefix ) | ( (u64)( blockIndex + 1 ) << 32 );
			cur = atomicCAS( g_iterator, expected, newValue );
			globalPrefix = cur & 0xFFFFFFFF;
		} while( cur != expected );

		gp = globalPrefix;
	}

	__syncthreads();

	u32 globalPrefix = gp;

	for( int i = 0; i < CLASSIC_SCAN_BLOCK_SIZE; i += CLASSIC_SCAN_THREADS_PER_BLOCK )
	{
		u32 itemIndex = blockIndex * CLASSIC_SCAN_BLOCK_SIZE + i + threadIdx.x;
		smem[threadIdx.x] = itemIndex < numberOfInputs ? inout[itemIndex] : 0;

		__syncthreads();

		globalPrefix += scanExclusive<u32>( globalPrefix, smem, CLASSIC_SCAN_THREADS_PER_BLOCK );

		if( itemIndex < numberOfInputs )
		{
			inout[itemIndex] = smem[threadIdx.x];
		}

		__syncthreads();
	}
}




template <bool keyPair>
__device__ __forceinline__ void classic_reorder( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, u32 numberOfInputs, u32* scanBuffer, u32 startBits, u32 iteration )
{
	__shared__ u32 pSum[BIN_SIZE];

	struct SMem
	{
		struct Phase1
		{
			u16 blockHistogram[BIN_SIZE];
			u16 lpSum[BIN_SIZE * REORDER_NUMBER_OF_WARPS];
		};
		struct Phase2
		{
			RADIX_SORT_KEY_TYPE elements[RADIX_SORT_BLOCK_SIZE];
		};
		struct Phase3
		{
			RADIX_SORT_VALUE_TYPE elements[RADIX_SORT_BLOCK_SIZE];
			u8 buckets[RADIX_SORT_BLOCK_SIZE];
		};

		union
		{
			Phase1 phase1;
			Phase2 phase2;
			Phase3 phase3;
		} u;
	};
	__shared__ SMem smem;

	u32 bitLocation = startBits + N_RADIX * iteration;
	u32 blockIndex = blockIdx.x;
	u32 numberOfBlocks = div_round_up( numberOfInputs, RADIX_SORT_BLOCK_SIZE );

	for( int bucketIndex = threadIdx.x; bucketIndex < BIN_SIZE; bucketIndex += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 counterIndex = bucketIndex * numberOfBlocks + blockIndex;
		pSum[bucketIndex] = scanBuffer[counterIndex];
	}

	clearShared<BIN_SIZE * REORDER_NUMBER_OF_WARPS, REORDER_NUMBER_OF_THREADS_PER_BLOCK, u16>( smem.u.phase1.lpSum, 0 );

	__syncthreads();

	RADIX_SORT_KEY_TYPE keys[REORDER_NUMBER_OF_ITEM_PER_THREAD];
	u32 localOffsets[REORDER_NUMBER_OF_ITEM_PER_THREAD];

	int warp = threadIdx.x / WARP_SIZE;
	int lane = threadIdx.x % WARP_SIZE;

	for( int i = 0, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i + lane;
		if( itemIndex < numberOfInputs )
		{
			keys[k] = inputKeys[itemIndex];
		}
	}
	for( int i = 0, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i + lane;
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );

		// check the attendees
		u32 broThreads =
#if defined( ITS )
			__ballot_sync( 0xFFFFFFFF,
#else
			__ballot(
#endif
						   itemIndex < numberOfInputs );

		for( int j = 0; j < N_RADIX; ++j )
		{
			u32 bit = ( bucketIndex >> j ) & 0x1;
			u32 difference = ( 0xFFFFFFFF * bit ) ^
#if defined( ITS )
							 __ballot_sync( 0xFFFFFFFF, bit != 0 );
#else
							 __ballot( bit != 0 );
#endif
			broThreads &= ~difference;
		}

		u32 lowerMask = ( 1u << lane ) - 1;
		auto digitCount = smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
		localOffsets[k] = digitCount + __popc( broThreads & lowerMask );

#if defined( ITS )
		__syncwarp( 0xFFFFFFFF );
#else
		__syncthreads();
#endif
		u32 leaderIdx = __ffs( broThreads ) - 1;
		if( lane == leaderIdx )
		{
			smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp] = digitCount + __popc( broThreads );
		}
#if defined( ITS )
		__syncwarp( 0xFFFFFFFF );
#else
		__syncthreads();
#endif
	}

	__syncthreads();

	for( int bucketIndex = threadIdx.x; bucketIndex < BIN_SIZE; bucketIndex += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = 0;
		for( int warp = 0; warp < REORDER_NUMBER_OF_WARPS; warp++ )
		{
			s += smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
		}
		smem.u.phase1.blockHistogram[bucketIndex] = s;
	}

	__syncthreads();

	u32 prefix = 0;
	for( int i = 0; i < BIN_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		prefix += scanExclusive<u16>( prefix, smem.u.phase1.blockHistogram + i, min( REORDER_NUMBER_OF_THREADS_PER_BLOCK, BIN_SIZE ) );
	}

	for( int bucketIndex = threadIdx.x; bucketIndex < BIN_SIZE; bucketIndex += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 s = smem.u.phase1.blockHistogram[bucketIndex];

		pSum[bucketIndex] -= s; // pre-substruct to avoid pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex] to calculate destinations

		for( int w = 0; w < REORDER_NUMBER_OF_WARPS; w++ )
		{
			int index = bucketIndex * REORDER_NUMBER_OF_WARPS + w;
			u32 n = smem.u.phase1.lpSum[index];
			smem.u.phase1.lpSum[index] = s;
			s += n;
		}
	}

	__syncthreads();

	for( int k = 0; k < REORDER_NUMBER_OF_ITEM_PER_THREAD; k++ )
	{
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
		localOffsets[k] += smem.u.phase1.lpSum[bucketIndex * REORDER_NUMBER_OF_WARPS + warp];
	}

	__syncthreads();

	for( int i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;
		u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
		if( itemIndex < numberOfInputs )
		{
			smem.u.phase2.elements[localOffsets[k]] = keys[k];
		}
	}

	__syncthreads();

	for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
	{
		u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
		if( itemIndex < numberOfInputs )
		{
			auto item = smem.u.phase2.elements[i];
			u32 bucketIndex = extractDigit( getKeyBits( item ), bitLocation );

			// u32 dstIndex = pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex];
			u32 dstIndex = pSum[bucketIndex] + i;
			outputKeys[dstIndex] = item;
		}
	}

	if constexpr( keyPair )
	{
		__syncthreads();

		for( int i = lane, k = 0; i < REORDER_NUMBER_OF_ITEM_PER_WARP; i += WARP_SIZE, k++ )
		{
			u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + warp * REORDER_NUMBER_OF_ITEM_PER_WARP + i;
			u32 bucketIndex = extractDigit( getKeyBits( keys[k] ), bitLocation );
			if( itemIndex < numberOfInputs )
			{
				smem.u.phase3.elements[localOffsets[k]] = inputValues[itemIndex];
				smem.u.phase3.buckets[localOffsets[k]] = bucketIndex;
			}
		}

		__syncthreads();

		for( int i = threadIdx.x; i < RADIX_SORT_BLOCK_SIZE; i += REORDER_NUMBER_OF_THREADS_PER_BLOCK )
		{
			u32 itemIndex = blockIndex * RADIX_SORT_BLOCK_SIZE + i;
			if( itemIndex < numberOfInputs )
			{
				auto item = smem.u.phase3.elements[i];
				u32 bucketIndex = smem.u.phase3.buckets[i];

				// u32 dstIndex = pSum[bucketIndex] + i - smem.u.phase1.blockHistogram[bucketIndex];
				u32 dstIndex = pSum[bucketIndex] + i;
				outputValues[dstIndex] = item;
			}
		}
	}
}
extern "C" __global__ void __launch_bounds__( REORDER_NUMBER_OF_THREADS_PER_BLOCK ) classic_reorderKey( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, u32 numberOfInputs, u32* scanBuffer, u32 startBits, u32 iteration )
{
	classic_reorder<false /*keyPair*/>( inputKeys, outputKeys, nullptr, nullptr, numberOfInputs, scanBuffer, startBits, iteration );
}
extern "C" __global__ void __launch_bounds__( REORDER_NUMBER_OF_THREADS_PER_BLOCK ) classic_reorderKeyPair( RADIX_SORT_KEY_TYPE* inputKeys, RADIX_SORT_KEY_TYPE* outputKeys, RADIX_SORT_VALUE_TYPE* inputValues, RADIX_SORT_VALUE_TYPE* outputValues, u32 numberOfInputs, u32* scanBuffer, u32 startBits, u32 iteration )
{
	classic_reorder<true /*keyPair*/>( inputKeys, outputKeys, inputValues, outputValues, numberOfInputs, scanBuffer, startBits, iteration );
}