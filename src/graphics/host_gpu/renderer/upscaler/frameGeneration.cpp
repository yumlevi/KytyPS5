#include "graphics/host_gpu/renderer/upscaler/frameGeneration.h"

#include "common/assert.h"
#include "common/driverAddressSpace.h"
#include "common/liveControl.h"
#include "common/logging/log.h"
#include "gpu_blit_shaders/gpu_framegen_motion_spv.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#endif

#ifdef KYTY_FRAMEGEN_STREAMLINE
// (after windows.h)
#include <softpub.h>
#include <wincrypt.h>
#include <wintrust.h>
// NVIDIA Streamline SDK headers (_Build/deps/streamline/sdk/include, see CMakeLists.txt).
#include <sl.h>
#include <sl_consts.h>
#include <sl_dlss_g.h>
#include <sl_helpers.h>
#include <sl_pcl.h>
#include <sl_reflex.h>

#include <filesystem>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>
#endif

namespace Libs::Graphics::FrameGen {

namespace {

// DLSS-G frames' Reflex sleep (slReflexSleep, every present): 0 skips it.
Common::LiveControl::Switch g_reflex_sleep("fg_reflex_sleep", 1);
// 0: the camera is never taken from the game's view constants (a comparison).
Common::LiveControl::Switch g_use_camera("fg_camera", 1);

struct Sources {
	ImageId       depth, motion;
	uint64_t      depth_address = 0, motion_address = 0;
	ImageViewInfo motion_view; // the game's own view of its velocity image, as a 2D image
	View          view;
	bool          valid = false;
};

int         g_generate = 0;     // frames to generate per rendered frame (0 = off)
bool        g_ready    = false; // Streamline loaded and DLSS-G available on the device
bool        g_for_upscaler = false; // the emulator's upscaler takes the inputs too (CaptureForUpscaler)
bool        g_untaken      = false; // renderer thread: captured, and not taken by the upscaler yet
Sources     g_sources;          // renderer thread: the dispatch FindInputs noted, until it is recorded
FrameInputs g_capture;          // renderer thread: the current guest frame's inputs, until its flip
uint64_t    g_guest_frame = 0;  // renderer thread
std::atomic<uint32_t> g_presented_frames {1};
std::string           g_vulkan_library;

// One line for the user (the console) and the log.
[[gnu::format(printf, 1, 2)]] void Say(const char* format, ...) {
	char    text[1024];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	std::printf("Frame generation: %s\n", text);
	std::fflush(stdout);
	LOGF("frame generation: %s\n", text);
}

bool HasStencil(vk::Format format) {
	return format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD24UnormS8Uint ||
	       format == vk::Format::eD32SfloatS8Uint;
}

bool IsDepthFormat(vk::Format format) {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32Sfloat:
		case vk::Format::eD32SfloatS8Uint:
		case vk::Format::eX8D24UnormPack32: return true;
		default: return false;
	}
}

void Transition(vk::CommandBuffer command, VulkanImage& image, vk::ImageAspectFlags aspect,
                vk::ImageLayout layout, vk::AccessFlags2 access, vk::PipelineStageFlags2 stage) {
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask        = image.state.pl_stage;
	barrier.srcAccessMask       = image.state.access_mask;
	barrier.dstStageMask        = stage;
	barrier.dstAccessMask       = access;
	barrier.oldLayout           = image.state.layout;
	barrier.newLayout           = layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image.image;
	barrier.subresourceRange    = {aspect, 0, 1, 0, 1};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
}

// (Re)creates `image` as a sampled image of `format` and `extent` that is also used as `usage`,
// with its view.
void Configure(GraphicContext& graphics, VulkanImage& image, vk::ImageView& view, vk::Format format,
               vk::Extent3D extent, vk::ImageUsageFlags usage) {
	if (image.image != nullptr && image.format == format && image.extent.width == extent.width &&
	    image.extent.height == extent.height) {
		return;
	}
	if (view != nullptr) {
		graphics.device.destroyImageView(view, nullptr);
		view = nullptr;
	}
	if (image.image != nullptr) {
		graphics.DeleteImage(image);
	}
	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = usage | vk::ImageUsageFlagBits::eSampled;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, image, MemoryPriority::Class::Pinned)) {
		EXIT("frame generation: cannot allocate a %ux%u input image\n", extent.width, extent.height);
	}
	vk::ImageViewCreateInfo view_info {};
	view_info.image            = image.image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = format;
	view_info.subresourceRange = {IsDepthFormat(format) ? vk::ImageAspectFlagBits::eDepth
	                                                    : vk::ImageAspectFlagBits::eColor,
	                              0, 1, 0, 1};
	RequireVulkanSuccess(graphics.device.createImageView(&view_info, nullptr, &view),
	                     "frame generation input view");
}

// The top left `extent` of `source` into `target`.
void CopyInto(vk::CommandBuffer command, Image& source, VulkanImage& target, vk::Extent2D extent) {
	const auto aspect = IsDepthFormat(source.backing.format) ? vk::ImageAspectFlagBits::eDepth
	                                                         : vk::ImageAspectFlagBits::eColor;
	// (Copied is the depth; a barrier on an image with stencil too takes both.)
	const vk::ImageAspectFlags whole =
	    HasStencil(target.format) ? vk::ImageAspectFlags(aspect) | vk::ImageAspectFlagBits::eStencil
	                              : vk::ImageAspectFlags(aspect);
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transition(command, target, whole, vk::ImageLayout::eTransferDstOptimal,
	           vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eTransfer);
	vk::ImageCopy copy {};
	copy.srcSubresource = {aspect, 0, 0, 1};
	copy.dstSubresource = {aspect, 0, 0, 1};
	copy.extent         = vk::Extent3D {extent.width, extent.height, 1};
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, target.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
	Transition(command, target, whole, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}

// The motion vectors of `source` (the game's packed velocity image, gpu_framegen_motion.comp)
// into the top left `extent` of `target`.
void DecodeMotion(GraphicContext& graphics, vk::CommandBuffer command, vk::ImageView source,
                  VulkanImage& target, vk::ImageView target_view, vk::Extent2D extent) {
	// (Made once, on the renderer thread; they last as long as the device.)
	static vk::DescriptorSetLayout descriptor_layout = nullptr;
	static vk::PipelineLayout      pipeline_layout   = nullptr;
	static vk::Pipeline            pipeline          = nullptr;
	if (pipeline == nullptr) {
		const std::array bindings {
		    vk::DescriptorSetLayoutBinding {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}};
		vk::DescriptorSetLayoutCreateInfo descriptor_info {};
		descriptor_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptor_info.bindingCount = static_cast<uint32_t>(bindings.size());
		descriptor_info.pBindings    = bindings.data();
		RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr, &descriptor_layout),
		                     "frame generation descriptor layout");
		const vk::PushConstantRange  push {vk::ShaderStageFlagBits::eCompute, 0, 2 * sizeof(uint32_t)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount         = 1;
		layout_info.pSetLayouts            = &descriptor_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push;
		RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout_info, nullptr, &pipeline_layout),
		                     "frame generation pipeline layout");
		pipeline = CreateComputePipeline(graphics.device, GPU_FRAMEGEN_MOTION_SPV, pipeline_layout,
		                                 "frame generation motion pipeline");
	}
	Transition(command, target, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
	           vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader);
	const std::array images {vk::DescriptorImageInfo {nullptr, source, vk::ImageLayout::eGeneral},
	                         vk::DescriptorImageInfo {nullptr, target_view, vk::ImageLayout::eGeneral}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo      = &images[i];
	}
	const uint32_t size[] {extent.width, extent.height};
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, static_cast<uint32_t>(writes.size()),
	                             writes.data());
	command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	command.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(size), size);
	command.dispatch((extent.width + 7u) / 8u, (extent.height + 7u) / 8u, 1);
	Transition(command, target, vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eShaderReadOnlyOptimal,
	           vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
}

// ---------------------------------------------------------------------------------------------
// The game's side. Unreal Engine 5's Temporal Super Resolution, as Saros (PPSA07631) runs it:
// its first pass ("dilate velocity") samples the scene's velocity (16_16_16_16 UNORM, image 0)
// and depth (the depth target as 32_FLOAT, image 1) and writes, beside depths (16_16_FLOAT,
// image 2), the frame's motion vectors to a 32_UINT image (image 4): for every pixel the
// velocity of the nearest of its neighbours, the camera's motion included, in the engine's
// 16-bit encoding, x in the low half and y in the high half (decoded by DecodeMotion). The
// images have the size the engine allocates for its highest render resolution; the frame is in
// their top left part (dynamic resolution).
// Its buffers: 0 the upscaler's constants (FTSRCommonParameters), 1 the engine's view constants
// (FViewUniformShaderParameters). Rows are float4s.
// ---------------------------------------------------------------------------------------------
namespace Tsr {
constexpr uint32_t ViewportRect = 2;  // int4: the frame's rectangle in the images (min, max)
constexpr uint32_t ViewportSize = 3;  // float4: its size and the inverse
constexpr uint32_t Jitter       = 16; // float2: the sample's offset in pixels (y down); int: camera cut
constexpr uint32_t Rows         = 18;
} // namespace Tsr
namespace UeView {
constexpr uint32_t ViewToClipNoAA = 32;  // 4 rows
constexpr uint32_t WorldOrigin    = 72;  // float3: the camera's position
constexpr uint32_t Forward        = 73;  // float3 each: forward, up, right
constexpr uint32_t ClipToPrevClip = 136; // 4 rows, without jitter
constexpr uint32_t FieldOfView    = 146; // float2: horizontal and vertical, radians
constexpr uint32_t ViewSize       = 148; // float4: the frame's size and the inverse
constexpr uint32_t PreExposure    = 155; // an int, a float, then the pre-exposure and its inverse
constexpr uint32_t Rows           = 149;
} // namespace UeView

struct Rows {
	const float* data = nullptr;
	uint32_t     rows = 0;

	[[nodiscard]] float At(uint32_t row, uint32_t column) const { return data[row * 4u + column]; }
	[[nodiscard]] uint32_t Word(uint32_t row, uint32_t column) const {
		uint32_t word = 0;
		std::memcpy(&word, data + row * 4u + column, sizeof(word));
		return word;
	}
};

// A buffer of float4 rows in guest memory (constants the game's CPU wrote).
Rows RowsOf(const ShaderRecompiler::IR::DescriptorValue& descriptor) {
	const auto buffer = DecodeNativeDescriptor<ShaderBufferResource>(descriptor);
	if (buffer.Base48() == 0 || buffer.Stride() != 16u) {
		return {};
	}
	return {reinterpret_cast<const float*>(buffer.Base48()), buffer.NumRecords()};
}

bool NearlyEqual(float a, float b) {
	return std::fabs(a - b) <= 1e-4f * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

// The frame's view from the upscaler's and the engine's constants; false when the upscaler's are
// not what this build of the game has (then the dispatch is not taken at all).
bool ReadView(const ShaderRecompiler::IR::ResourceSnapshot& resources, vk::Extent2D images, View& view) {
	const auto tsr = RowsOf(resources.buffers[0]);
	if (tsr.rows < Tsr::Rows) {
		return false;
	}
	const uint32_t x0 = tsr.Word(Tsr::ViewportRect, 0), y0 = tsr.Word(Tsr::ViewportRect, 1);
	const uint32_t x1 = tsr.Word(Tsr::ViewportRect, 2), y1 = tsr.Word(Tsr::ViewportRect, 3);
	if (x0 != 0 || y0 != 0 || x1 == 0 || y1 == 0 || x1 > images.width || y1 > images.height ||
	    tsr.At(Tsr::ViewportSize, 0) != static_cast<float>(x1) ||
	    tsr.At(Tsr::ViewportSize, 1) != static_cast<float>(y1)) {
		return false;
	}
	view.width  = x1;
	view.height = y1;
	view.jitter = {tsr.At(Tsr::Jitter, 0), tsr.At(Tsr::Jitter, 1)};
	view.cut    = tsr.Word(Tsr::Jitter, 2) != 0;
	if (!std::isfinite(view.jitter[0]) || !std::isfinite(view.jitter[1]) || std::fabs(view.jitter[0]) > 1.0f ||
	    std::fabs(view.jitter[1]) > 1.0f) {
		view.jitter = {0.0f, 0.0f};
	}
	// (DecodeMotion stores the way to the previous position in parts of the frame.)
	view.motion_scale = {1.0f, 1.0f};

	// The camera, when the view constants are laid out as this build's: a projection without
	// jitter (reversed depth, infinite far plane: w = view z, z = near plane) and the frame's size.
	view.camera = false;
	if (resources.buffers.size() < 2) {
		return true;
	}
	const auto ue = RowsOf(resources.buffers[1]);
	if (ue.rows < UeView::Rows) {
		return true;
	}
	// The factor the scene's color is stored with (the engine divides it out again when it tone
	// maps): taken when its inverse stands next to it, as this build's constants have it.
	if (ue.rows > UeView::PreExposure) {
		const float factor = ue.At(UeView::PreExposure, 2), inverse = ue.At(UeView::PreExposure, 3);
		if (std::isfinite(factor) && factor > 1e-6f && factor < 1e6f && NearlyEqual(factor * inverse, 1.0f)) {
			view.pre_exposure = factor;
		}
	}
	if (!g_use_camera.On()) {
		return true;
	}
	const auto m = [&](uint32_t row, uint32_t column) { return ue.At(UeView::ViewToClipNoAA + row, column); };
	const bool projection = m(0, 0) > 0.0f && m(0, 1) == 0.0f && m(0, 2) == 0.0f && m(0, 3) == 0.0f && m(1, 0) == 0.0f &&
	                        m(1, 1) > 0.0f && m(1, 2) == 0.0f && m(1, 3) == 0.0f && m(2, 0) == 0.0f && m(2, 1) == 0.0f &&
	                        m(2, 2) == 0.0f && m(2, 3) == 1.0f && m(3, 0) == 0.0f && m(3, 1) == 0.0f && m(3, 2) > 0.0f &&
	                        m(3, 3) == 0.0f;
	const bool size = ue.At(UeView::ViewSize, 0) == static_cast<float>(x1) && ue.At(UeView::ViewSize, 1) == static_cast<float>(y1);
	const float fov = ue.At(UeView::FieldOfView, 1);
	if (!projection || !size || !(fov > 0.01f && fov < 3.1f) || !NearlyEqual(std::tan(fov * 0.5f) * m(1, 1), 1.0f)) {
		return true;
	}
	for (uint32_t i = 0; i < 16; ++i) {
		view.view_to_clip[i]      = ue.At(UeView::ViewToClipNoAA + i / 4u, i % 4u);
		view.clip_to_prev_clip[i] = ue.At(UeView::ClipToPrevClip + i / 4u, i % 4u);
		if (!std::isfinite(view.clip_to_prev_clip[i])) {
			return true;
		}
	}
	for (uint32_t i = 0; i < 3; ++i) {
		view.position[i] = ue.At(UeView::WorldOrigin, i);
		view.forward[i]  = ue.At(UeView::Forward, i);
		view.up[i]       = ue.At(UeView::Forward + 1, i);
		view.right[i]    = ue.At(UeView::Forward + 2, i);
	}
	view.near_plane   = m(3, 2);
	view.vertical_fov = fov;
	view.camera       = true;
	return true;
}

#ifdef KYTY_FRAMEGEN_STREAMLINE
// Streamline entry points (sl.interposer.dll, loaded at run time) and feature functions.
struct Api {
	PFun_slInit*               init                 = nullptr;
	PFun_slIsFeatureSupported* is_feature_supported = nullptr;
	PFun_slIsFeatureLoaded*    is_feature_loaded    = nullptr;
	PFun_slGetFeatureFunction* get_feature_function = nullptr;
	PFun_slGetNewFrameToken*   get_new_frame_token  = nullptr;
	PFun_slSetConstants*       set_constants        = nullptr;
	PFun_slSetTagForFrame*     set_tag_for_frame    = nullptr;
	PFun_slDLSSGSetOptions*    dlssg_set_options    = nullptr;
	PFun_slDLSSGGetState*      dlssg_get_state      = nullptr;
	PFun_slReflexSetOptions*   reflex_set_options   = nullptr;
	PFun_slPCLSetMarker*       pcl_set_marker       = nullptr;
	PFun_slReflexSleep*        reflex_sleep         = nullptr;
};
Api             g_api;
sl::FrameToken* g_token       = nullptr; // presenting thread: the frame being presented
uint32_t        g_frame_index = 0;
bool            g_options_set = false;
vk::Extent2D    g_options_backbuffer {};
vk::Extent2D    g_options_images {};
vk::Format      g_options_format = vk::Format::eUndefined; // SDR or HDR10 swapchain
uint64_t        g_presents       = 0;
// DLSS-G on (generating) or off with its resources kept (eRetainResourcesWhenOff): off after
// TurnOffAfter presents in a row without inputs (menus, movies, loading), on after TurnOnAfter
// with them, so a fade or a cut does not flap it. On with null tags it would be asked to generate
// frames it has no inputs for.
constexpr uint32_t TurnOffAfter = 30, TurnOnAfter = 10;
bool               g_mode_on    = false;
uint32_t           g_run_with = 0, g_run_without = 0; // presents in a row with inputs, without
// A frame with inputs that was never shown (NoteDroppedFrame): the next one's motion spans two
// game frames.
std::atomic<bool> g_discontinuity {false};

// Matrices in Streamline's convention: row-major, row vectors (p' = p * M).
using Mat4 = sl::float4x4;

Mat4 Matrix(const std::array<float, 16>& m) {
	Mat4 r;
	for (uint32_t i = 0; i < 4; ++i) {
		r[i] = sl::float4(m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]);
	}
	return r;
}

float At(const sl::float4& v, uint32_t i) {
	return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

bool Inverted(const Mat4& m, Mat4& out) {
	double a[4][8] {};
	for (uint32_t i = 0; i < 4; ++i) {
		for (uint32_t j = 0; j < 4; ++j) a[i][j] = At(m[i], j);
		a[i][4 + i] = 1.0;
	}
	for (uint32_t c = 0; c < 4; ++c) {
		uint32_t pivot = c;
		for (uint32_t r = c + 1; r < 4; ++r) {
			if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) pivot = r;
		}
		if (std::fabs(a[pivot][c]) < 1e-12) return false;
		for (uint32_t j = 0; j < 8; ++j) std::swap(a[c][j], a[pivot][j]);
		const double scale = 1.0 / a[c][c];
		for (uint32_t j = 0; j < 8; ++j) a[c][j] *= scale;
		for (uint32_t r = 0; r < 4; ++r) {
			if (r == c) continue;
			const double f = a[r][c];
			for (uint32_t j = 0; j < 8; ++j) a[r][j] -= f * a[c][j];
		}
	}
	for (uint32_t i = 0; i < 4; ++i) {
		out[i] = sl::float4(static_cast<float>(a[i][4]), static_cast<float>(a[i][5]),
		                    static_cast<float>(a[i][6]), static_cast<float>(a[i][7]));
	}
	return true;
}

// How many times this error has come (its text after Streamline's time stamp and thread).
uint64_t ErrorRepeats(const char* message) {
	std::string_view text(message);
	if (const auto stamp = text.find("us]"); stamp != std::string_view::npos) text.remove_prefix(stamp + 3);
	static std::mutex                             mutex;
	static std::unordered_map<uint64_t, uint64_t> counts;
	const uint64_t                                key = std::hash<std::string_view> {}(text);
	std::lock_guard                               lock(mutex);
	return ++counts[key];
}

void LogMessage(sl::LogType type, const char* message) {
	LOGF("Streamline[%d]: %s", static_cast<int>(type), message);
	// (Streamline speaks while the device is made and its plugins start: where a start that dies
	// inside NVIDIA's libraries stops is the last thing the log can say.)
	Log::Flush();
	// KYTY_FRAMEGEN_VERBOSE: every message on stdout, not only errors.
	static const bool verbose = std::getenv("KYTY_FRAMEGEN_VERBOSE") != nullptr;
	if (type != sl::LogType::eError && !verbose) return;
	// The same error again: the first 3 printed, then every 1000th with its count (a tag refused
	// every frame is a hundred lines a second).
	const uint64_t repeats = type == sl::LogType::eError && !verbose && message != nullptr ? ErrorRepeats(message) : 1;
	if (repeats <= 3) {
		std::printf("Streamline %s: %s", type == sl::LogType::eError ? "error" : "log", message);
		if (repeats == 3) std::printf("Streamline: that error 3 times; from now on every 1000th of it is printed\n");
	} else if (repeats % 1000 == 0) {
		std::printf("Streamline error (%" PRIu64 " times): %s", repeats, message);
	} else {
		return;
	}
	std::fflush(stdout);
}

template <typename F>
bool FeatureFunction(sl::Feature feature, const char* name, F*& function) {
	void* pointer = nullptr;
	if (g_api.get_feature_function(feature, name, pointer) != sl::Result::eOk || pointer == nullptr) {
		LOGF("frame generation: %s is unavailable\n", name);
		return false;
	}
	function = reinterpret_cast<F*>(pointer);
	return true;
}

// The file's first signature verifies and its signer is NVIDIA.
bool IsNvidiaSigned(const std::wstring& path) {
	WINTRUST_FILE_INFO file {};
	file.cbStruct      = sizeof(file);
	file.pcwszFilePath = path.c_str();
	WINTRUST_DATA data {};
	data.cbStruct            = sizeof(data);
	data.dwUIChoice          = WTD_UI_NONE;
	data.fdwRevocationChecks = WTD_REVOKE_NONE;
	data.dwUnionChoice       = WTD_CHOICE_FILE;
	data.pFile               = &file;
	data.dwStateAction       = WTD_STATEACTION_VERIFY;
	data.dwProvFlags         = WTD_CACHE_ONLY_URL_RETRIEVAL;
	GUID       action        = WINTRUST_ACTION_GENERIC_VERIFY_V2;
	const LONG status        = WinVerifyTrust(static_cast<HWND>(INVALID_HANDLE_VALUE), &action, &data);
	bool       nvidia        = false;
	if (status == ERROR_SUCCESS) {
		// The signer: the leaf certificate of the first signature's chain.
		auto* provider = WTHelperProvDataFromStateData(data.hWVTStateData);
		auto* signer   = provider != nullptr ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
		auto* cert     = signer != nullptr ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
		wchar_t name[256] {};
		if (cert != nullptr && cert->pCert != nullptr &&
		    CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name, 256) > 1)
			nvidia = std::wstring(name) == L"NVIDIA Corporation";
	}
	data.dwStateAction = WTD_STATEACTION_CLOSE;
	(void)WinVerifyTrust(static_cast<HWND>(INVALID_HANDLE_VALUE), &action, &data);
	return nvidia;
}

sl::Resource TaggedImage(const VulkanImage& image, vk::ImageView view) {
	sl::Resource resource(sl::ResourceType::eTex2d, static_cast<VkImage>(image.image), nullptr,
	                      static_cast<VkImageView>(view), static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
	resource.width        = image.extent.width;
	resource.height       = image.extent.height;
	resource.nativeFormat = static_cast<uint32_t>(image.format);
	resource.mipLevels    = 1;
	resource.arrayLayers  = 1;
	resource.flags        = 0;
	resource.usage        = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image.usage));
	return resource;
}
#endif

} // namespace

bool Enabled() {
	return g_ready;
}

bool Requested() {
	return g_generate > 0;
}

const char* VulkanLibrary() {
	return g_vulkan_library.empty() ? nullptr : g_vulkan_library.c_str();
}

uint32_t PresentedFrames() {
	return g_presented_frames.load(std::memory_order_relaxed);
}

std::string SupportProblem(vk::PhysicalDevice physical_device) {
	if (g_generate == 0) {
		return {};
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// (Streamline answers before the device too: its programming guide asks while the adapters
	// are enumerated. A refused device says nothing itself.)
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
	const auto  result       = g_api.is_feature_supported(sl::kFeatureDLSS_G, adapter);
	std::string problem;
	switch (result) {
		case sl::Result::eOk: break;
		case sl::Result::eErrorOSDisabledHWS:
			problem = "Windows' hardware-accelerated GPU scheduling is off (Settings > System > Display > Graphics > "
			          "Advanced graphics settings; then restart the PC)";
			break;
		case sl::Result::eErrorOSOutOfDate: problem = "this Windows version is too old for DLSS frame generation"; break;
		case sl::Result::eErrorDriverOutOfDate: problem = "the NVIDIA driver is too old for DLSS frame generation"; break;
		case sl::Result::eErrorNoSupportedAdapterFound:
		case sl::Result::eErrorAdapterNotSupported:
			problem = "this graphics card has no DLSS frame generation (NVIDIA RTX 40 or newer)";
			break;
		default: problem = std::string("Streamline does not offer DLSS frame generation here (") + sl::getResultAsStr(result) + ")"; break;
	}
	Say("%s", problem.empty() ? "Streamline offers DLSS-G with this GPU, driver and Windows" : problem.c_str());
	return problem;
#else
	(void)physical_device;
	return {};
#endif
}

void BypassCommandHooks(vk::Device device) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// KYTY_FRAMEGEN_HOOKS=1 keeps Streamline's wrappers (for comparison).
	if (g_generate == 0 || std::getenv("KYTY_FRAMEGEN_HOOKS") != nullptr) {
		return;
	}
	// The real loader: Streamline itself forwards to it.
	HMODULE loader          = GetModuleHandleW(L"vulkan-1.dll");
	auto*   get_device_proc = loader != nullptr ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(reinterpret_cast<void*>(
	                                                  GetProcAddress(loader, "vkGetDeviceProcAddr")))
	                                            : nullptr;
	if (get_device_proc == nullptr) {
		return;
	}
	auto&      dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto native     = static_cast<VkDevice>(device);
	uint32_t   replaced   = 0;
	const auto direct     = [&](auto& function, const char* name) {
		if (auto* pointer = get_device_proc(native, name); pointer != nullptr) {
			function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(pointer);
			++replaced;
		}
	};
	direct(dispatcher.vkBeginCommandBuffer, "vkBeginCommandBuffer");
	direct(dispatcher.vkCmdBindPipeline, "vkCmdBindPipeline");
	direct(dispatcher.vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
	direct(dispatcher.vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
	LOGF("frame generation: %u command-buffer calls bypass Streamline\n", replaced);
#else
	(void)device;
#endif
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// From the guest's ceiling (kernel/memoryAddressSpace.inc, HostUserMax() + 1) to 2^40.
constexpr uint64_t DriverSpaceEnd = Common::DriverAddressSpace::End;
// What is kept aside: one block, or the free pieces when something sat in the range already
// (a library another program put into the process at its start). No heap here: nothing may
// map memory before the hold.
static std::array<void*, 64> g_driver_blocks {};
static uint32_t              g_driver_block_count = 0;
static uint64_t              g_driver_held        = 0;
static bool                  g_driver_hold_asked  = false;
#endif

void HoldDriverAddressSpace(bool upscaler) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const char* value = std::getenv("KYTY_FRAMEGEN");
	if (!(upscaler || (value != nullptr && std::atoi(value) > 0)) || g_driver_hold_asked) {
		return;
	}
	g_driver_hold_asked = true;
	// The guest's ceiling comes down with it (before the guest's memory is set up), 64 GB
	// below 1 TB where 16 are left otherwise. KYTY_DRIVER_SPACE_GB=<16 to 128> chooses.
	const char* room = std::getenv("KYTY_DRIVER_SPACE_GB");
	Common::DriverAddressSpace::SetRoomGiB(room != nullptr && room[0] != '\0'
	                                           ? std::strtoull(room, nullptr, 10)
	                                           : Common::DriverAddressSpace::DefaultRoomGiB);
	const uint64_t DriverSpaceBase = Common::DriverAddressSpace::Base();
	if (void* whole = VirtualAlloc(reinterpret_cast<void*>(DriverSpaceBase), DriverSpaceEnd - DriverSpaceBase,
	                               MEM_RESERVE, PAGE_NOACCESS);
	    whole != nullptr) {
		g_driver_blocks[g_driver_block_count++] = whole;
		g_driver_held                           = DriverSpaceEnd - DriverSpaceBase;
		return;
	}
	for (uint64_t at = DriverSpaceBase; at < DriverSpaceEnd && g_driver_block_count < g_driver_blocks.size();) {
		MEMORY_BASIC_INFORMATION region {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &region, sizeof(region)) != sizeof(region)) {
			break;
		}
		const auto begin = reinterpret_cast<uint64_t>(region.BaseAddress);
		const auto end   = std::min<uint64_t>(begin + region.RegionSize, DriverSpaceEnd);
		// (Reservations start at multiples of 64 KiB.)
		const auto from = (std::max(begin, at) + 0xffffull) & ~0xffffull;
		if (region.State == MEM_FREE && from < end) {
			if (void* piece = VirtualAlloc(reinterpret_cast<void*>(from), end - from, MEM_RESERVE, PAGE_NOACCESS);
			    piece != nullptr) {
				g_driver_blocks[g_driver_block_count++] = piece;
				g_driver_held += end - from;
			}
		}
		at = std::max(end, at + 0x10000ull);
	}
#else
	(void)upscaler;
#endif
}

void ReleaseDriverAddressSpace() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	for (uint32_t index = 0; index < g_driver_block_count; ++index) {
		VirtualFree(g_driver_blocks[index], 0, MEM_RELEASE);
	}
	g_driver_block_count = 0;
#endif
}

DriverSpace MeasureDriverAddressSpace() {
	DriverSpace space {};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	space.asked = g_driver_hold_asked;
	space.held  = g_driver_held;
	const auto walk = [](uint64_t from, uint64_t to, uint64_t* free, uint64_t* largest) {
		uint64_t run = 0;
		for (uint64_t at = from; at < to;) {
			MEMORY_BASIC_INFORMATION region {};
			if (VirtualQuery(reinterpret_cast<const void*>(at), &region, sizeof(region)) != sizeof(region)) {
				break;
			}
			const auto begin = reinterpret_cast<uint64_t>(region.BaseAddress);
			const auto end   = std::min<uint64_t>(begin + region.RegionSize, to);
			if (region.State == MEM_FREE) {
				*free += end - std::max(begin, at);
				run += end - std::max(begin, at);
				*largest = std::max(*largest, run);
			} else {
				run = 0;
			}
			at = std::max(end, at + 0x1000ull);
		}
	};
	const uint64_t DriverSpaceBase = Common::DriverAddressSpace::Base();
	space.room                     = DriverSpaceEnd - DriverSpaceBase;
	walk(DriverSpaceBase, DriverSpaceEnd, &space.free_top, &space.largest_top);
	walk(0x10000ull, DriverSpaceBase, &space.free_below, &space.largest_below);
#endif
	return space;
}

void Initialize() {
	const char* value = std::getenv("KYTY_FRAMEGEN");
	g_generate        = value != nullptr ? std::clamp(std::atoi(value), 0, 5) : 0;
	if (g_generate == 0) {
		return;
	}
	// (Every way out without Streamline turns it off.)
	struct Off {
		bool done = false;
		~Off() {
			if (!done) g_generate = 0;
		}
	} off;
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// NVIDIA's Streamline runtime, which the emulator does not ship (the user's own copy: the
	// Streamline SDK's bin\x64): KYTY_STREAMLINE_DIR, else a "streamline" folder next to the
	// emulator, else the SDK unpacked beside the build (_Build\deps\streamline\sdk). A folder may
	// be the SDK's own (its bin\x64 is taken).
	std::vector<std::filesystem::path> candidates;
	wchar_t                            exe[MAX_PATH] {};
	GetModuleFileNameW(nullptr, exe, MAX_PATH);
	const auto exe_dir = std::filesystem::path(exe).parent_path();
	// (Wide: a folder's name in any language.)
	if (const wchar_t* env = _wgetenv(L"KYTY_STREAMLINE_DIR"); env != nullptr && *env != L'\0') {
		candidates.emplace_back(env);
	} else {
		candidates.push_back(exe_dir / "streamline");
		candidates.push_back(exe_dir.parent_path() / "deps" / "streamline" / "sdk");
	}
	std::filesystem::path dir;
	for (const auto& candidate: candidates) {
		for (const auto& folder: {candidate, candidate / "bin" / "x64"}) {
			std::error_code error;
			if (dir.empty() && std::filesystem::exists(folder / "sl.interposer.dll", error)) dir = folder;
		}
	}
	if (dir.empty()) {
		Say("NVIDIA's Streamline runtime was not found (sl.interposer.dll: KYTY_STREAMLINE_DIR, or a streamline "
		    "folder next to the emulator)");
		return;
	}
	const auto interposer = dir / "sl.interposer.dll";
	// Only NVIDIA's own (Streamline then checks its plugins' signatures itself).
	if (!IsNvidiaSigned(interposer.wstring())) {
		Say("%s is not signed by NVIDIA, not loaded", interposer.string().c_str());
		return;
	}
	HMODULE module = LoadLibraryW(interposer.c_str());
	if (module == nullptr) {
		Say("cannot load %s", interposer.string().c_str());
		return;
	}
	const auto load = [&](auto& function, const char* name) {
		function = reinterpret_cast<std::remove_reference_t<decltype(function)>>(
		    reinterpret_cast<void*>(GetProcAddress(module, name)));
		return function != nullptr;
	};
	if (!load(g_api.init, "slInit") || !load(g_api.is_feature_supported, "slIsFeatureSupported") ||
	    !load(g_api.is_feature_loaded, "slIsFeatureLoaded") || !load(g_api.get_feature_function, "slGetFeatureFunction") ||
	    !load(g_api.get_new_frame_token, "slGetNewFrameToken") || !load(g_api.set_constants, "slSetConstants") ||
	    !load(g_api.set_tag_for_frame, "slSetTagForFrame")) {
		Say("sl.interposer.dll lacks an entry point");
		return;
	}
	static const std::wstring       plugin_dir = dir.wstring();
	static const wchar_t*           plugin_paths[] {plugin_dir.c_str()};
	static std::vector<sl::Feature> features {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL};
	sl::Preferences                 preferences {};
	preferences.showConsole        = false;
	preferences.logLevel           = std::getenv("KYTY_FRAMEGEN_VERBOSE") ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
	preferences.pathsToPlugins     = plugin_paths;
	preferences.numPathsToPlugins  = 1;
	preferences.pathToLogsAndData  = nullptr;
	preferences.logMessageCallback = LogMessage;
	preferences.flags              = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
	preferences.featuresToLoad     = features.data();
	preferences.numFeaturesToLoad  = static_cast<uint32_t>(features.size());
	preferences.engine             = sl::EngineType::eCustom;
	preferences.engineVersion      = "1.0";
	preferences.projectId          = "3e0f7a52-9c1b-4d6e-8a2f-5b7c9d1e2f30";
	preferences.renderAPI          = sl::RenderAPI::eVulkan;
	if (const auto result = g_api.init(preferences, sl::kSDKVersion); result != sl::Result::eOk) {
		Say("slInit failed (%d)", static_cast<int>(result));
		return;
	}
	// SDL loads the interposer as the Vulkan loader (the window sets its hint from
	// VulkanLibrary): every instance, device and swapchain call then goes through Streamline,
	// which adds what DLSS-G needs.
	g_vulkan_library = interposer.string();
	// (Streamline's DLLs do not survive ExitProcess: common/assert.h.)
	Common::ExitWithoutDetach();
	LOGF("frame generation: Streamline loaded from %s, %d generated frame(s)\n", dir.string().c_str(), g_generate);
	off.done = true;
#else
	Say("this build has no Streamline support (its SDK was not under _Build/deps/streamline/sdk when it was configured)");
#endif
}

void OnDevice(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
	(void)instance;
	(void)device;
	if (g_generate == 0) {
		return;
	}
#ifdef KYTY_FRAMEGEN_STREAMLINE
	sl::AdapterInfo adapter {};
	adapter.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
	bool loaded              = false;
	if (g_api.is_feature_supported(sl::kFeatureDLSS_G, adapter) != sl::Result::eOk ||
	    g_api.is_feature_loaded(sl::kFeatureDLSS_G, loaded) != sl::Result::eOk || !loaded) {
		Say("DLSS-G is not supported on this device");
		return;
	}
	if (!FeatureFunction(sl::kFeatureDLSS_G, "slDLSSGSetOptions", g_api.dlssg_set_options) ||
	    !FeatureFunction(sl::kFeatureDLSS_G, "slDLSSGGetState", g_api.dlssg_get_state) ||
	    !FeatureFunction(sl::kFeatureReflex, "slReflexSetOptions", g_api.reflex_set_options) ||
	    !FeatureFunction(sl::kFeaturePCL, "slPCLSetMarker", g_api.pcl_set_marker)) {
		return;
	}
	// (Required with DLSS-G, whatever the Reflex mode: Streamline's DLSS-G and Reflex guides.)
	(void)FeatureFunction(sl::kFeatureReflex, "slReflexSleep", g_api.reflex_sleep);
	sl::ReflexOptions reflex {};
	reflex.mode = sl::ReflexMode::eLowLatency;
	if (const auto result = g_api.reflex_set_options(reflex); result != sl::Result::eOk) {
		LOGF("frame generation: slReflexSetOptions failed (%d)\n", static_cast<int>(result));
	}
	g_ready = true;
	Say("DLSS-G ready (%d generated frame(s) per frame)", g_generate);
#else
	(void)physical_device;
#endif
}

void BeforePresent(const FrameInputs* inputs, bool repeated, vk::Extent2D backbuffer, vk::Rect2D region,
                   vk::Format format, uint32_t image_count) {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready) {
		return;
	}
	const sl::ViewportHandle viewport {0};
	const bool               usable = !repeated && inputs != nullptr && inputs->valid;
	if (!repeated) {
		g_run_with    = usable ? g_run_with + 1 : 0;
		g_run_without = usable ? 0 : g_run_without + 1;
	}
	const bool         mode_on = g_mode_on ? g_run_without < TurnOffAfter : g_run_with >= TurnOnAfter;
	const vk::Extent2D images  = usable ? vk::Extent2D {inputs->images->motion.extent.width,
	                                                    inputs->images->motion.extent.height}
	                                    : g_options_images;
	// DLSS-G options: on while frames carry inputs; sizes are hints for its allocations.
	if (!g_options_set || g_options_format != format || mode_on != g_mode_on || g_options_backbuffer != backbuffer ||
	    g_options_images != images) {
		sl::DLSSGOptions options {};
		options.mode                = mode_on ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
		options.flags               = sl::DLSSGFlags::eRetainResourcesWhenOff;
		options.numFramesToGenerate = static_cast<uint32_t>(g_generate);
		options.numBackBuffers      = image_count;
		options.colorWidth          = backbuffer.width;
		options.colorHeight         = backbuffer.height;
		options.colorBufferFormat   = static_cast<uint32_t>(format);
		if (usable) {
			options.mvecDepthWidth    = images.width;
			options.mvecDepthHeight   = images.height;
			options.mvecBufferFormat  = static_cast<uint32_t>(inputs->images->motion.format);
			options.depthBufferFormat = static_cast<uint32_t>(inputs->images->depth.format);
		}
		if (const auto result = g_api.dlssg_set_options(viewport, options); result != sl::Result::eOk) {
			LOGF("frame generation: slDLSSGSetOptions failed (%d)\n", static_cast<int>(result));
		}
		g_options_set        = true;
		g_options_backbuffer = backbuffer;
		g_options_format     = format;
		g_options_images     = images;
		if (mode_on != g_mode_on) {
			if (mode_on) Say("DLSS-G on");
			else Say("DLSS-G off: %u frames without the game's upscaler (menus, movies, loading)", TurnOffAfter);
			g_mode_on = mode_on;
		}
	}

	++g_frame_index;
	if (g_api.get_new_frame_token(g_token, &g_frame_index) != sl::Result::eOk || g_token == nullptr) {
		g_token = nullptr;
		return;
	}
	auto& token = *g_token;
	// The frame's Reflex sleep, then its markers (all at its present: the game's frame was made
	// before).
	if (g_api.reflex_sleep != nullptr && g_reflex_sleep.On()) g_api.reflex_sleep(token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationStart, token);
	g_api.pcl_set_marker(sl::PCLMarker::eSimulationEnd, token);
	g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitStart, token);

	sl::Constants constants {};
	Mat4          identity;
	for (uint32_t i = 0; i < 4; ++i) {
		identity[i] = sl::float4(i == 0, i == 1, i == 2, i == 3);
	}
	// Without the game's camera: a projection like most games' (a 60 degree view, reversed
	// depth) and no camera motion beyond what the vectors carry.
	Mat4 projection;
	projection[0] = sl::float4(0.974279f, 0, 0, 0);
	projection[1] = sl::float4(0, 1.732051f, 0, 0);
	projection[2] = sl::float4(0, 0, 0, 1);
	projection[3] = sl::float4(0, 0, 0.1f, 0);
	constants.cameraViewToClip = projection;
	Inverted(projection, constants.clipToCameraView);
	constants.clipToLensClip      = identity;
	constants.clipToPrevClip      = identity;
	constants.prevClipToClip      = identity;
	constants.jitterOffset        = sl::float2(0.0f, 0.0f);
	constants.mvecScale           = sl::float2(1.0f, 1.0f);
	constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
	constants.cameraPos           = sl::float3(0, 0, 0);
	constants.cameraUp            = sl::float3(0, 1, 0);
	constants.cameraRight         = sl::float3(1, 0, 0);
	constants.cameraFwd           = sl::float3(0, 0, 1);
	constants.cameraNear          = 0.1f;
	constants.cameraFar           = 100000.0f;
	constants.cameraFOV           = 1.047198f;
	constants.cameraAspectRatio   = 16.0f / 9.0f;
	constants.motionVectorsInvalidValue = sl::INVALID_FLOAT;
	constants.depthInverted             = sl::Boolean::eTrue;
	constants.cameraMotionIncluded      = sl::Boolean::eTrue;
	constants.motionVectors3D           = sl::Boolean::eFalse;
	// (A frame left unshown before this one: its motion is to the game's previous frame, not to
	// the one presented last.)
	const bool discontinuity = g_discontinuity.exchange(false, std::memory_order_relaxed) || (usable && inputs->view.cut);
	constants.reset                  = usable && !discontinuity ? sl::Boolean::eFalse : sl::Boolean::eTrue;
	constants.orthographicProjection = sl::Boolean::eFalse;
	constants.motionVectorsDilated   = sl::Boolean::eTrue;
	constants.motionVectorsJittered  = sl::Boolean::eFalse;
	if (usable) {
		const auto& view       = inputs->view;
		constants.mvecScale    = sl::float2(view.motion_scale[0], view.motion_scale[1]);
		constants.jitterOffset = sl::float2(view.jitter[0], view.jitter[1]);
		if (view.camera) {
			constants.cameraViewToClip = Matrix(view.view_to_clip);
			Inverted(constants.cameraViewToClip, constants.clipToCameraView);
			constants.clipToPrevClip = Matrix(view.clip_to_prev_clip);
			if (!Inverted(constants.clipToPrevClip, constants.prevClipToClip)) {
				constants.clipToPrevClip = identity;
				constants.prevClipToClip = identity;
			}
			constants.cameraPos         = sl::float3(view.position[0], view.position[1], view.position[2]);
			constants.cameraUp          = sl::float3(view.up[0], view.up[1], view.up[2]);
			constants.cameraRight       = sl::float3(view.right[0], view.right[1], view.right[2]);
			constants.cameraFwd         = sl::float3(view.forward[0], view.forward[1], view.forward[2]);
			constants.cameraNear        = view.near_plane;
			constants.cameraFOV         = view.vertical_fov;
			constants.cameraAspectRatio = view.view_to_clip[5] / view.view_to_clip[0];
		}
	}
	if (const auto result = g_api.set_constants(constants, token, viewport); result != sl::Result::eOk && g_presents < 8) {
		LOGF("frame generation: slSetConstants failed (%d)\n", static_cast<int>(result));
	}

	if (usable) {
		sl::Resource     depth  = TaggedImage(inputs->images->depth, inputs->images->depth_view);
		sl::Resource     motion = TaggedImage(inputs->images->motion, inputs->images->motion_view);
		const sl::Extent frame {0, 0, inputs->view.width, inputs->view.height};
		const sl::Extent game {static_cast<uint32_t>(region.offset.y), static_cast<uint32_t>(region.offset.x),
		                       region.extent.width, region.extent.height};
		// The backbuffer is tagged only when the game image covers part of it.
		const bool partial = region.offset.x != 0 || region.offset.y != 0 || region.extent.width != backbuffer.width ||
		                     region.extent.height != backbuffer.height;
		sl::ResourceTag tags[] {
		    sl::ResourceTag(&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent, &frame),
		    sl::ResourceTag(&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent, &frame),
		    sl::ResourceTag(nullptr, sl::kBufferTypeBackbuffer, sl::ResourceLifecycle::eValidUntilPresent, &game),
		};
		if (const auto result = g_api.set_tag_for_frame(token, viewport, tags, partial ? 3u : 2u, nullptr);
		    result != sl::Result::eOk && g_presents < 8) {
			LOGF("frame generation: slSetTagForFrame failed (%d)\n", static_cast<int>(result));
		}
	} else {
		// Menus, movies, loading: no depth and motion (DLSS-G then presents the frame as it is).
		sl::ResourceTag tags[] {
		    sl::ResourceTag(nullptr, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent),
		    sl::ResourceTag(nullptr, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent),
		};
		g_api.set_tag_for_frame(token, viewport, tags, static_cast<uint32_t>(std::size(tags)), nullptr);
	}
	g_api.pcl_set_marker(sl::PCLMarker::eRenderSubmitEnd, token);
	g_api.pcl_set_marker(sl::PCLMarker::ePresentStart, token);
#else
	(void)inputs;
	(void)repeated;
	(void)backbuffer;
	(void)region;
	(void)format;
	(void)image_count;
#endif
}

void AfterPresent() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	if (!g_ready || g_token == nullptr) {
		return;
	}
	g_api.pcl_set_marker(sl::PCLMarker::ePresentEnd, *g_token);
	const auto presents = ++g_presents;
	if (presents % 30 != 1) {
		return;
	}
	sl::DLSSGState state {};
	if (g_api.dlssg_get_state(sl::ViewportHandle {0}, state, nullptr) != sl::Result::eOk) {
		return;
	}
	g_presented_frames.store(std::max(state.numFramesActuallyPresented, 1u), std::memory_order_relaxed);
	// More generated frames asked for than the GPU makes (3x and up need an RTX 50): as many as
	// it makes.
	if (state.numFramesToGenerateMax != 0 && static_cast<uint32_t>(g_generate) > state.numFramesToGenerateMax) {
		Say("%d generated frame(s) asked for, the GPU makes up to %u", g_generate, state.numFramesToGenerateMax);
		g_generate    = static_cast<int>(state.numFramesToGenerateMax);
		g_options_set = false;
	}
	if (presents % 600 == 1) {
		LOGF("frame generation: status=0x%x presented=%u max_generated=%u vram=%" PRIu64 " MiB\n",
		     static_cast<uint32_t>(state.status), state.numFramesActuallyPresented, state.numFramesToGenerateMax,
		     state.estimatedVRAMUsageInBytes >> 20u);
	}
#endif
}

void SwapchainChanging() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	// Off before the swapchain is made again (DLSS-G guide: before a resize or a full screen
	// change).
	if (!g_ready || !g_mode_on) return;
	sl::DLSSGOptions options {};
	options.mode                = sl::DLSSGMode::eOff;
	options.flags               = sl::DLSSGFlags::eRetainResourcesWhenOff;
	options.numFramesToGenerate = static_cast<uint32_t>(g_generate);
	if (const auto result = g_api.dlssg_set_options(sl::ViewportHandle {0}, options); result != sl::Result::eOk)
		LOGF("frame generation: slDLSSGSetOptions (off) failed (%d)\n", static_cast<int>(result));
	g_mode_on     = false;
	g_run_with    = 0;
	g_options_set = false; // (the next present sends its sizes again)
#endif
}

void NoteDroppedFrame() {
#ifdef KYTY_FRAMEGEN_STREAMLINE
	g_discontinuity.store(true, std::memory_order_relaxed);
#endif
	static std::atomic<uint32_t> dropped {0};
	if (const auto count = dropped.fetch_add(1, std::memory_order_relaxed) + 1; count <= 8 || count % 600 == 0)
		LOGF("frame generation: %u frame(s) with inputs left unshown (the next resets DLSS-G's history)\n", count);
}

void FindInputs(uint64_t shader_hash, const ShaderRecompiler::IR::ResourceSnapshot& resources,
                const PreparedBindings& bindings, std::span<const uint32_t, 3> groups) {
	constexpr uint32_t Velocity = 0, Depth = 1, Depths = 2, Motion = 4;
	if (!CapturesInputs() || resources.images.size() <= Motion || bindings.images.size() <= Motion ||
	    resources.buffers.empty()) {
		return;
	}
	const auto velocity = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[Velocity]);
	const auto depth    = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[Depth]);
	const auto depths   = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[Depths]);
	const auto motion   = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[Motion]);
	if (velocity.Format() != Prospero::BufferFormat::k16_16_16_16UNorm || depth.Format() != Prospero::BufferFormat::k32Float ||
	    depths.Format() != Prospero::BufferFormat::k16_16Float || motion.Format() != Prospero::BufferFormat::k32UInt ||
	    depth.Width5() != motion.Width5() || depth.Height5() != motion.Height5() || velocity.Width5() != motion.Width5() ||
	    velocity.Height5() != motion.Height5()) {
		return;
	}
	View view;
	if (!ReadView(resources, {static_cast<uint32_t>(motion.Width5()) + 1u, static_cast<uint32_t>(motion.Height5()) + 1u}, view)) {
		return;
	}
	if (view.camera) {
		// (The path a session took, for its log and for a replay of it.)
		Common::LiveControl::NoteCamera(view.position.data(), view.forward.data());
	}
	// (The game's view of its velocity image, of its first layer as a 2D image.)
	auto motion_view        = bindings.images[Motion].desc.view_info;
	motion_view.type        = vk::ImageViewType::e2D;
	motion_view.layer_count = 1;
	motion_view.level_count = 1;
	motion_view.usage       = vk::ImageUsageFlagBits::eStorage;
	g_sources = {bindings.images[Depth].image_id,
	             bindings.images[Motion].image_id,
	             bindings.images[Depth].desc.info.data.address,
	             bindings.images[Motion].desc.info.data.address,
	             motion_view,
	             view,
	             true};
	// (Once per shader and size: what was found.)
	static uint64_t logged_hash = 0;
	static uint32_t logged_size = 0;
	if (logged_hash != shader_hash || logged_size != (view.width << 16u | view.height)) {
		logged_hash = shader_hash;
		logged_size = view.width << 16u | view.height;
		LOGF("frame generation: inputs from CS 0x%016" PRIx64 " (%ux%ux%u groups): depth 0x%010" PRIx64
		     ", motion 0x%010" PRIx64 ", frame %ux%u in %ux%u images, camera %s\n",
		     shader_hash, groups[0], groups[1], groups[2], g_sources.depth_address, g_sources.motion_address, view.width,
		     view.height, static_cast<uint32_t>(motion.Width5()) + 1u, static_cast<uint32_t>(motion.Height5()) + 1u,
		     view.camera ? "from the view constants" : "unknown");
	}
}

void CaptureInputs(GraphicContext& graphics, TextureCache& cache, CommandBuffer& command) {
	if (!g_sources.valid) {
		return;
	}
	const auto sources = g_sources;
	g_sources.valid    = false;
	auto& depth        = cache.GetImage(sources.depth);
	auto& motion       = cache.GetImage(sources.motion);
	if (!depth.registered || !motion.registered || depth.backing.image == nullptr || motion.backing.image == nullptr ||
	    depth.info.data.address != sources.depth_address || motion.info.data.address != sources.motion_address ||
	    depth.backing.extent.width != motion.backing.extent.width ||
	    depth.backing.extent.height != motion.backing.extent.height || sources.view.width > motion.backing.extent.width ||
	    sources.view.height > motion.backing.extent.height) {
		return;
	}
	if (g_capture.images == nullptr) {
		g_capture.images = std::make_unique<Images>();
	}
	auto& images = *g_capture.images;
	Configure(graphics, images.depth, images.depth_view, depth.backing.format, depth.backing.extent,
	          vk::ImageUsageFlagBits::eTransferDst);
	Configure(graphics, images.motion, images.motion_view, vk::Format::eR16G16Sfloat, motion.backing.extent,
	          vk::ImageUsageFlagBits::eStorage);
	const auto motion_source = motion.FindView(sources.motion_view);
	command.EndRendering();
	// (The decode binds its own pipeline: the next guest dispatch or draw binds its own again.)
	auto               vk_command = command.PipelineHandle();
	const vk::Extent2D frame {sources.view.width, sources.view.height};
	CopyInto(vk_command, depth, images.depth, frame);
	motion.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
	               ImageSubresourceRange {sources.motion_view.base_level, 1, sources.motion_view.base_layer, 1}, vk_command);
	DecodeMotion(graphics, vk_command, motion_source, images.motion, images.motion_view, frame);
	g_capture.view  = sources.view;
	g_capture.valid = true;
	g_untaken       = true;
}

void CaptureForUpscaler() {
	g_for_upscaler = true;
}

bool CapturesInputs() {
	return g_ready || g_for_upscaler;
}

// The last pass of Unreal Engine 5's Temporal Super Resolution ("update history"), as Saros runs
// it: it samples the frame's color (11_11_10_FLOAT, the scene with its translucency, written by
// the pass before at the render images' size) and its history, a 11_11_10_FLOAT array at the
// output size whose layers are the last frames' pictures, and writes the frame's picture into
// one layer of that same array. It is found by those images, not by the shader.
std::optional<UpscalerPass> FindUpscalerPass(const ShaderRecompiler::IR::ResourceSnapshot&         resources,
                                             std::span<const ShaderRecompiler::IR::ImageResource> images,
                                             const PreparedBindings&                               bindings) {
	if (!g_for_upscaler || !g_untaken || !g_capture.valid || g_capture.images == nullptr) {
		return std::nullopt;
	}
	const auto     size  = g_capture.images->motion.extent;
	const uint32_t count = static_cast<uint32_t>(std::min({resources.images.size(), images.size(), bindings.images.size()}));
	uint32_t       color = count, output = count, reduced = UpscalerPass::NoImage;
	bool           one_reduced = true;
	for (uint32_t i = 0; i < count; ++i) {
		const auto image = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
		if (image.IsNull() || image.Format() != Prospero::BufferFormat::k11_11_10Float) {
			continue;
		}
		const uint32_t width = static_cast<uint32_t>(image.Width5()) + 1u, height = static_cast<uint32_t>(image.Height5()) + 1u;
		if (images[i].written) {
			if (width >= size.width && height >= size.height && (width > size.width || height > size.height)) {
				if (output != count) {
					return std::nullopt;
				}
				output = i;
			} else {
				// (A small copy of the picture, a quarter of its size in Saros, for the exposure.)
				one_reduced = reduced == UpscalerPass::NoImage;
				reduced     = i;
			}
		} else if (width == size.width && height == size.height) {
			if (color != count) {
				return std::nullopt;
			}
			color = i;
		}
	}
	if (color == count || output == count) {
		return std::nullopt;
	}
	// The history it reads is the array it writes.
	const auto address = bindings.images[output].desc.info.data.address;
	bool       history = false;
	for (uint32_t i = 0; i < count; ++i) {
		history = history || (i != output && !images[i].written && bindings.images[i].desc.info.data.address == address);
	}
	if (!history || bindings.images[color].desc.info.data.address == address) {
		return std::nullopt;
	}
	return UpscalerPass {color, output, one_reduced ? reduced : UpscalerPass::NoImage, g_capture.view.width,
	                     g_capture.view.height};
}

FrameInputs* TakeCaptured() {
	if (!g_untaken || !g_capture.valid || g_capture.images == nullptr) {
		return nullptr;
	}
	g_untaken = false;
	return &g_capture;
}

void SetUpscalerLayout(vk::CommandBuffer command, Images& images, bool upscaling) {
	const auto layout = upscaling ? vk::ImageLayout::eGeneral : vk::ImageLayout::eShaderReadOnlyOptimal;
	const vk::ImageAspectFlags depth =
	    HasStencil(images.depth.format) ? vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil
	                                    : vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth);
	Transition(command, images.depth, depth, layout, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eAllCommands);
	Transition(command, images.motion, vk::ImageAspectFlagBits::eColor, layout, vk::AccessFlagBits2::eShaderRead,
	           vk::PipelineStageFlagBits2::eAllCommands);
}

void PrepareInputs(FrameInputs& inputs) {
	inputs.valid = false;
	++g_guest_frame;
	if (!g_ready || !g_capture.valid) {
		return;
	}
	// (The frame's last images, shown long ago, take the next capture.)
	std::swap(inputs.images, g_capture.images);
	inputs.view        = g_capture.view;
	inputs.guest_frame = g_guest_frame;
	inputs.valid       = true;
	g_capture.valid    = false;
}

void DestroyInputs(GraphicContext& graphics, FrameInputs& inputs) {
	inputs.valid = false;
	if (inputs.images == nullptr) {
		return;
	}
	for (auto* view: {&inputs.images->depth_view, &inputs.images->motion_view}) {
		if (*view != nullptr) {
			graphics.device.destroyImageView(*view, nullptr);
			*view = nullptr;
		}
	}
	for (auto* image: {&inputs.images->depth, &inputs.images->motion}) {
		if (image->image != nullptr) {
			graphics.DeleteImage(*image);
		}
	}
	inputs.images.reset();
}

} // namespace Libs::Graphics::FrameGen
