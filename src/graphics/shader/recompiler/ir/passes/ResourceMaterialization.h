#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		// V# stride layout; the stride field is 1 for any nonzero stride (the stride itself is
		// read at run time, see FillBufferStrides).
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		// A direct V# the renderer binds as the null buffer (no base or no records), where every
		// access is out of bounds. It never changes the module; SpecializationServes uses it.
		bool                   unbound                         = false;
		// indirect_mapping_offset is a plan-fixed flattened-SRT word holding the key map's
		// offset, so the module does not depend on where the scene's key maps end up.
		bool                   indirect_mapping_header         = false;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		Prospero::BufferFormat        srgb_decode                = Prospero::BufferFormat::kInvalid;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          skip_degamma               = false;
		bool                          fmask                      = false;
		bool                          heap                       = false;
		// Resolved at run time through the bindless translation table (no candidates).
		bool                          bindless                   = false;
		// The bindless key is the byte offset of the T# in its heap's records (an image heap the
		// shader indexes at a scalar offset); the translation index is the key divided by four.
		bool                          bindless_byte_keys             = false;
		// See Buffer::indirect_mapping_header.
		bool                          indirect_mapping_header        = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset = 0;
		uint32_t indirect_search_iterations = 0;
		// See Buffer::indirect_mapping_header.
		bool indirect_mapping_header          = false;
		bool operator==(const Sampler&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
// The plan's walker loads guest memory as `loads` says (see CompileSrtWalker). Extraction is
// deterministic: two plans extracted from one program are interchangeable. A deferred plan has no
// walker yet; CompileSrtWalker(plan, {}, loads) adds it (e.g. on another thread) before any walk.
enum class SrtWalkerCompilation { Now, Deferred };
ResourcePlan ExtractResourcePlan(const Program&       program,
                                 SrtWalkerLoads       loads       = SrtWalkerLoads::Direct,
                                 SrtWalkerCompilation compilation = SrtWalkerCompilation::Now);

// Shader cache record of a plan; loading it into a default-constructed plan rebuilds the walker
// and scratch state like extraction. A failed load leaves a plan that must not be used.
// `loads` and `compilation` are ExtractResourcePlan's: the bytes hold neither, so one record
// gives the plan of any of them.
std::vector<uint8_t> SaveResourcePlan(const ResourcePlan& plan);
bool                 LoadResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan,
                                      SrtWalkerLoads       loads       = SrtWalkerLoads::Direct,
                                      SrtWalkerCompilation compilation = SrtWalkerCompilation::Now);

// The user-data slots (SGPR minus the plan's user_data_base) the plan's values read, one bit each
// for slots below 64; a slot at or above 64 sets every bit.
[[nodiscard]] uint64_t PlanUserDataReads(const ResourcePlan& plan);

// Refreshes cached resources and specialization in place. A failed refresh must not be used;
// snapshot.unavailable tells a failure caused by the draw's guest data from an unsupported shader.
// Answers whether a range overlaps any of a list of (address, bytes) ranges, in logarithmic
// time per query after one sort: materialization checks every GPU-selected store candidate (up
// to thousands) against every read a table walk recorded (up to tens of thousands). The index
// is built into `index` on the first query; later queries must pass the same list.
class RangeOverlapIndex {
public:
	[[nodiscard]] bool Overlaps(std::span<const std::pair<uint64_t, uint64_t>> ranges,
	                            std::vector<std::pair<uint64_t, uint64_t>>& index, uint64_t base,
	                            uint64_t size);

private:
	bool m_built = false;
};

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// How many specialization failures MaterializeResources has printed: one line per shader per
// session, none for a speculative walk (SrtRuntime::speculative).
uint64_t SpecializationFailureReports();
// The failure line this thread's last reporting MaterializeResources call printed, empty when
// none printed one since the last take.
std::string TakeSpecializationFailureReport();

// Whether a module compiled for `compiled` behaves exactly as one compiled for `current`
// would. Equal layouts do; so does any layout of an unbound direct buffer when both layouts read
// zero out of bounds, since an unbound buffer is bound so that every access is out of bounds
// (loads read the out-of-bounds value, stores and atomics are dropped). With `portable`, so does
// a layout whose buffer takes its format at run time for a buffer of one of the formats it
// selects from (PortableBufferFormats), and a layout whose storage image takes its store swizzle
// at run time (PortableImageSwizzle) for any swizzle: the same values, through a decoding
// selected per access, so a caller that has a module compiled for the format itself prefers that
// one (`portable` false). With the dispatch's `resources`, so does a layout whose image is
// declared float for raw loads and stores (RawClassImage) for a T# of an integer format that a
// float or UNorm view keeps (RawViewOf).
bool SpecializationServes(const ResourcePlan& program, const ResourceSpecialization& compiled,
                          const ResourceSpecialization& current, bool portable = true,
                          const ResourceSnapshot* resources = nullptr);

// Gives the unbound buffers of `target` the bound layouts of `donor` that serve them, so a
// module compiled for `target` also serves those bound cases.
void AdoptServingBuffers(const ResourcePlan& program, ResourceSpecialization& target,
                         const ResourceSpecialization& donor);

// Where an indirect root's key map starts in the snapshot's flattened SRT: its mapping offset, or
// the word its header holds (SIZE_MAX when the header is out of range).
size_t KeyMapOffset(const ResourceSnapshot& snapshot, const ResourceSpecialization::Buffer& root);
size_t KeyMapOffset(const ResourceSnapshot& snapshot, const ResourceSpecialization::Image& root);
size_t KeyMapOffset(const ResourceSnapshot& snapshot, const ResourceSpecialization::Sampler& root);

// Whether a module compiled for `compiled` serves the draw `snapshot` and `current` describe once
// its GPU-selected image candidates are moved into the module's candidate slots: every candidate
// a key selects needs a slot of its own layout (a null one any free slot), a key search needs at
// most the module's depth, and a root the draw does not map is served by a module that maps it
// (the draw's key map header names the empty map). Rewrites the snapshot's images and key map
// ordinals and `current` to the module's layout when it does; leaves both unchanged otherwise.
bool AdaptToSpecialization(const ResourcePlan& program, const ResourceSpecialization& compiled,
                           ResourceSnapshot& snapshot, ResourceSpecialization& current);

// Widens `current` to a layout that also serves the draws `prior` serves - per image root, the
// prior candidate slots plus the ones the draw has beyond them, the deeper key searches - and
// moves the draw into it. A shader whose tables change with the scene thereby compiles modules
// whose layouts only grow. Leaves both unchanged when the layouts do not combine.
// The candidate layout of `layout` at the key limit of its GPU-selected image tables. A table
// whose keys the host enumerates has at most as many keys as a table may have before it is
// resolved at run time instead, so its layouts stop growing there: every slot layout a mapped
// root has, once per key, and the key search of that many keys. A module compiled for it serves
// every draw whose tables hold candidates of those layouts (AdaptToSpecialization), so a table
// that grows with the scene takes one module instead of one per doubling (WidenSpecialization).
// Nullopt when `layout` has no such table or is at the limit already.
[[nodiscard]] std::optional<ResourceSpecialization>
WidestCandidateLayout(const ResourceSpecialization& layout);
// The layout the same program has once those tables hold more keys than the limit and are
// resolved at run time instead (bindless roots without candidate slots), for the programs it can
// be told for: searched image tables only, each with a plan-fixed header. A guess the precompile
// builds ahead; a draw uses the module only when its own layout is this one. Nullopt otherwise.
[[nodiscard]] std::optional<ResourceSpecialization>
BindlessTableLayout(const ResourceSpecialization& layout);
// Moves the draw's candidates into that layout, when the draw has such tables; false, with the
// draw as it was, when it has none or does not fit.
bool WidenToKeyLimit(const ResourcePlan& program, ResourceSnapshot& snapshot,
                     ResourceSpecialization& current);
void WidenSpecialization(const ResourcePlan& program, const ResourceSpecialization& prior,
                         ResourceSnapshot& snapshot, ResourceSpecialization& current);

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

// Kind of an image heap entry as root sees it, or 0 (and a zeroed value) when root reads it as
// null. Entries of one kind specialize identically.
uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value);
// The sampled resource that root resolves entries of kind as; kind 0 gives the identity of
// root's view of a heap.
ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind);
// Whether ApplyResourceSpecialization accepts `specialization` for a translated program: its
// resource counts and indirect roots fit the program's resources. A specialization recorded for
// another recompiler version may not.
bool SpecializationApplies(const Program& program, const ResourceSpecialization& specialization);
// The specialization a translated program gets from descriptors that are as plain as its code
// allows: direct buffers with a nonzero stride (formatted ones take their format at run time,
// see PortableBufferFormats; 32_UINT when the program has atomics on them), direct images of a
// float 2D format unless the code says otherwise, storage images with a run-time store swizzle
// (PortableImageSwizzle), no candidates. It needs no guest memory, so a
// pipeline can be built before the first dispatch; the shaders whose resources are plain get the
// module their dispatches use, or one that serves them (SpecializationServes). Empty when no
// descriptors can specialize the program's resources.
std::optional<ResourceSpecialization> CodeOnlySpecialization(const Program& program);

// Writes into the shader data the V# stride of every buffer binding with a runtime stride, the
// format word (the index of the V#'s format in PortableBufferFormats) of every buffer binding
// with a run-time format, and the image word of every portable image (PortableImageSwizzle):
// the T#'s DST_SEL and, for a raw-class image, the RawView of its format.
void FillBufferStrides(const ShaderInfo& info, const Bindings& bindings,
                       const ResourceSnapshot& snapshot, std::span<uint32_t> shader_data);
// Whether a run-time format module accesses a V# of this format and DST_SEL through its format
// functions (PortableBufferGenericFormat): any format with a decoding, with selectors that name
// a component, zero or one.
[[nodiscard]] bool PortableGenericBufferFormat(Prospero::BufferFormat format, uint32_t swizzle);
// Whether CodeOnlySpecialization gives formatted buffers a run-time format (the default),
// instead of 32_32_32_32_FLOAT.
void SetPortableBufferFormats(bool enabled);
[[nodiscard]] bool PortableBufferFormatsEnabled();
// For tests: a specialization made from descriptors also takes a run-time format for every
// buffer a portable module can access, so that whatever reads or writes such buffers runs the
// portable decoding.
void SetForcePortableBufferFormats(bool enabled);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
