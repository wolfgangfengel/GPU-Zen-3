#include <functional>
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>
#include <Orochi/GpuMemory.h>
#include "RadixSortConfigs.h"

namespace Radixsort
{
class ClassicSort final
{
public:
	using u32 = uint32_t;
	using u64 = uint64_t;

	struct KeyValueSoA
	{
		u32* key;
		u32* value;
	};

	ClassicSort( oroDevice device, OrochiUtils& oroutils, oroStream stream = 0 );

	// Allow move but disallow copy.
	ClassicSort( ClassicSort&& ) noexcept = default;
	ClassicSort& operator=( ClassicSort&& ) noexcept = default;
	ClassicSort( const ClassicSort& ) = delete;
	ClassicSort& operator=( const ClassicSort& ) = delete;
	~ClassicSort() = default;

	struct Callback
	{
		std::function<void( const char*, int )> kernelBeg = []( const char*, int ) {};
		std::function<void( const char*, int )> kernelEnd = []( const char*, int ) {};
	};

	void sort( const KeyValueSoA& elementsToSort, const KeyValueSoA& tmp, void* scanBuffer, uint32_t n, int startBit, int endBit, oroStream stream, Callback callback = Callback() ) noexcept;

	u32 scanBufferBytes( uint32_t n ) const;
private:
	// @brief Compile the kernels for radix sort.
	void compileKernels( ) noexcept;

	/// @brief Configure the settings, compile the kernels and allocate the memory.
	/// @param kernelPath The kernel path.
	/// @param includeDir The include directory.
	void configure( oroStream stream ) noexcept;

private:

	oroDevice m_device{};

	OrochiUtils& m_oroutils;

	oroFunction m_count;
	oroFunction m_scan;
	oroFunction m_reorderKey;
	oroFunction m_reorderKeyPair;

	Oro::GpuMemory<u64> m_scanIterator;
};

} // namespace Radixsort