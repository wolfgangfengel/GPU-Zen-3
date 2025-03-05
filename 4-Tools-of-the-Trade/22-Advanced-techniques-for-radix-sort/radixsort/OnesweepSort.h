#include <functional>
#include <Orochi/Orochi.h>
#include <Orochi/OrochiUtils.h>
#include "RadixSortConfigs.h"
#include <Orochi/GpuMemory.h>

namespace Radixsort
{
class OnesweepSort final
{
public:
	using u32 = uint32_t;
	using u64 = uint64_t;

	struct KeyValueSoA
	{
		u32* key;
		u32* value;
	};

	OnesweepSort( oroDevice device, OrochiUtils& oroutils, oroStream stream = 0 );

	// Allow move but disallow copy.
	OnesweepSort( OnesweepSort&& ) noexcept = default;
	OnesweepSort& operator=( OnesweepSort&& ) noexcept = default;
	OnesweepSort( const OnesweepSort& ) = delete;
	OnesweepSort& operator=( const OnesweepSort& ) = delete;
	~OnesweepSort() = default;

	struct Callback
	{
		std::function<void( const char*, int )> kernelBeg = []( const char*, int ) {};
		std::function<void( const char*, int )> kernelEnd = []( const char*, int ) {};
	};

	void sort( const KeyValueSoA& elementsToSort, const KeyValueSoA& tmp, uint32_t n, int startBit, int endBit, oroStream stream, Callback callback = Callback() ) noexcept;

private:
	// @brief Compile the kernels for radix sort.
	void compileKernels( ) noexcept;

	/// @brief Configure the settings, compile the kernels and allocate the memory.
	/// @param kernelPath The kernel path.
	/// @param includeDir The include directory.
	void configure( oroStream stream ) noexcept;

private:

	oroDevice m_device{};
	oroDeviceProp m_props{};

	OrochiUtils& m_oroutils;

	oroFunction m_count;
	oroFunction m_reorderKey;
	oroFunction m_reorderKeyPair;

	Oro::GpuMemory<uint8_t> m_lookbackBuffer;
	Oro::GpuMemory<uint8_t> m_gpSumBuffer;
	Oro::GpuMemory<u32> m_gpSumCounter;
	Oro::GpuMemory<u32> m_tailIterator;
};

} // namespace Radixsort